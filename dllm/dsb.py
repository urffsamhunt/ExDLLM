"""
Diffusion Schrödinger Bridge (DSB) for score-based training.

We model a Markov chain that transports a *pure input* datapoint DP1 to an
*output* datapoint DP2. The chain lives in a continuous embedding space so we
can write down an SDE.

Forward SDE (input -> output, corruption):
    dx_t = beta_t * (DP2 - x_t) dt + sqrt(beta_t) dW_t

The drift is the *relative position vector* of DP2 w.r.t. the current point
x_t, scaled by a time-dependent rate beta_t, plus Brownian noise. This is an
Ornstein-Uhlenbeck bridge pinned to both endpoints: it starts at DP1 (t=0) and
is pulled toward DP2 as t grows. Because the drift is linear in x and the
noise is additive Gaussian, the transition kernel is Gaussian in closed form:

    x_t | DP1, DP2  ~  N( mu_t, sigma_t^2 I )
    mu_t     = (1 - alpha_t) * DP2 + alpha_t * DP1
    alpha_t  = exp(-int_0^t beta_s ds)
    sigma_t^2 = (1 - exp(-2 int_0^t beta_s ds)) / 2

The exact conditional score is therefore available in closed form:

    s(x_t, t) = grad_{x_t} log p(x_t | DP1, DP2)
              = (mu_t - x_t) / sigma_t^2

This is the diffusion Schrödinger bridge: the forward process is pinned to
both endpoints, and the reverse process is learned by regressing a score
network against this closed-form conditional score (denoising score matching).

Reverse SDE (generation, DP1 -> DP2):
    dx_t = [ f(x_t, t) - sigma_t^2 * s_theta(x_t, t) ] dt + sqrt(beta_t) dW_t

where f(x_t, t) = beta_t * (DP2 - x_t) is the forward drift. At sampling time
DP2 is unknown, but it is recoverable from the learned score and the known
start DP1 (see `_estimate_target`), so the reverse drift still points along
the relative-position vector toward the output.

This module is self-contained: it does not depend on the discrete edit-based
DLLM model. It operates on arbitrary continuous vectors (token embeddings,
sentence embeddings, or raw features).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Noise schedules ───────────────────────────────────────────────────────────

def linear_beta_schedule(
    num_steps: int,
    beta_min: float = 0.0001,
    beta_max: float = 0.02,
) -> torch.Tensor:
    """Linearly spaced beta schedule over [beta_min, beta_max]."""
    return torch.linspace(beta_min, beta_max, num_steps)


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule (Nichol & Dhariwal), bounded away from 0."""
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps)
    alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


# ── Score network ─────────────────────────────────────────────────────────────

class MLPScoreNet(nn.Module):
    """
    A time-conditioned MLP score network.

    Maps (x, t) -> s_theta(x, t), an estimate of the (negative) conditional
    score grad_{x_t} log p(x_t | DP1, DP2). For a Brownian bridge the true
    score is (mu_t - x_t) / sigma_t^2, i.e. a displacement toward the target,
    so the network is a residual MLP over the concatenation of x and a
    Gaussian-Fourier time embedding.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        time_embed_dim: int = 128,
        cond_dim: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )

        layers = []
        in_dim = dim + cond_dim + time_embed_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, S, D) or (B, D); t: (B,) in [0, 1]
        # cond: optional conditioning with the same shape as x (e.g. DP1).
        t = t.reshape(-1, 1)
        t_emb = self.time_mlp(t)  # (B, time_embed_dim)
        # Broadcast time embedding across sequence dim if present.
        if x.dim() == 3:
            t_emb = t_emb.unsqueeze(1).expand(-1, x.shape[1], -1)
        h = torch.cat([x, t_emb], dim=-1)
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("score net built with cond_dim > 0 requires cond")
            h = torch.cat([cond, h], dim=-1)
        return self.net(h)


# ── The SDE / DSB bridge ──────────────────────────────────────────────────────

class DiffSchrodingerBridge(nn.Module):
    """
    A diffusion Schrödinger bridge from an input datapoint DP1 to an output
    datapoint DP2, trained by denoising score matching.

    Forward SDE (corruption):
        dx_t = beta_t * (DP2 - x_t) dt + sqrt(beta_t) dW_t

    The drift is the relative position vector of DP2 to the current point,
    plus Brownian noise. The exact conditional score is available in closed
    form (see module docstring), so training reduces to regressing a score
    network against it. Generation reverses the SDE from DP1 toward DP2.
    """

    def __init__(
        self,
        dim: int,
        score_net: Optional[nn.Module] = None,
        beta_schedule: str = "linear",
        num_steps: int = 1000,
        beta_min: float = 0.0001,
        beta_max: float = 0.02,
        condition_on_dp1: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_steps = num_steps
        # Condition the score net on the source endpoint DP1. Without this, the
        # best achievable drift target is the posterior mean E[DP2 | x_t], which
        # at low t collapses to E[DP2 | DP1] and puts a floor under
        # reconstruction_error. With conditioning, _estimate_target recovers
        # E[DP2 | x_t, DP1] and the floor drops to the noise realization.
        self.condition_on_dp1 = condition_on_dp1

        if beta_schedule == "linear":
            betas = linear_beta_schedule(num_steps, beta_min, beta_max)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(num_steps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        self.register_buffer("betas", betas)

        # B_t = int_0^t beta_s ds (Euler discretization: cumulative sum)
        self.register_buffer("B", torch.cumsum(betas, dim=0))
        # alpha_t = exp(-B_t), sigma_t^2 = (1 - exp(-2 B_t)) / 2
        self.register_buffer("alpha", torch.exp(-self.B))
        self.register_buffer("sigma2", (1.0 - torch.exp(-2.0 * self.B)) / 2.0)

        self.score_net = score_net or MLPScoreNet(dim=dim)

    betas: torch.Tensor
    B: torch.Tensor
    alpha: torch.Tensor
    sigma2: torch.Tensor
    score_net: nn.Module

    # ── Continuous-time schedule interpolation ────────────────────────────────

    def _interp(self, buf: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Linearly interpolate a schedule buffer at continuous t in [0, 1]."""
        idx = (t * (self.num_steps - 1)).clamp(0, self.num_steps - 1)
        lo = idx.floor().long()
        hi = (lo + 1).clamp(max=self.num_steps - 1)
        frac = (idx - lo).float()
        return buf[lo] * (1 - frac) + buf[hi] * frac

    def _alpha_at(self, t: torch.Tensor) -> torch.Tensor:
        return self._interp(self.alpha, t)

    def _sigma2_at(self, t: torch.Tensor) -> torch.Tensor:
        return self._interp(self.sigma2, t)

    def _beta_at(self, t: torch.Tensor) -> torch.Tensor:
        return self._interp(self.betas, t)

    def score_predict(self, x: torch.Tensor, t: torch.Tensor,
                      dp1: Optional[torch.Tensor] = None,
                      **kwargs) -> torch.Tensor:
        """Evaluate the score net, optionally conditioning on the source DP1."""
        cond = dp1 if self.condition_on_dp1 else None
        return self.score_net(x, t, cond=cond, **kwargs)

    # ── Forward diffusion: sample x_t from the bridge ────────────────────────

    def forward_sample(
        self,
        dp1: torch.Tensor,
        dp2: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample the noisy intermediate x_t given both endpoints.

        Args:
            dp1: (B, D) or (B, S, D) pure input.
            dp2: (B, D) or (B, S, D) output.
            t:   (B,) noise levels in [0, 1].

        Returns:
            x_t: noisy point.
            score: exact conditional score target (mu_t - x_t) / sigma_t^2.
        """
        t_b = t.reshape(-1, 1, 1) if dp1.dim() == 3 else t.reshape(-1, 1)
        alpha_t = self._alpha_at(t).reshape_as(t_b)
        sigma2_t = self._sigma2_at(t).reshape_as(t_b)

        mu = (1 - alpha_t) * dp2 + alpha_t * dp1  # linear bridge mean
        sigma_t = torch.sqrt(sigma2_t + 1e-8)
        z = torch.randn_like(dp1)
        x_t = mu + sigma_t * z
        score = (mu - x_t) / (sigma2_t + 1e-8)
        return x_t, score

    # ── Training loss: denoising score matching ───────────────────────────────

    def score_matching_loss(
        self,
        dp1: torch.Tensor,
        dp2: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Denoising score-matching loss for the bridge.

            L = E_t [ sigma_t^2 * || s_theta(x_t, t) - s(x_t | dp1, dp2) ||^2 ]

        The sigma^2 weighting is the standard denoising score-matching
        weighting that makes the objective a proper score-matching surrogate.

        Args:
            dp1: (B, D) or (B, S, D) input.
            dp2: (B, D) or (B, S, D) output.
            t:   optional (B,) noise levels; sampled uniformly if None.

        Returns:
            scalar loss.
        """
        B = dp1.shape[0]
        if t is None:
            t = torch.rand(B, device=dp1.device)
        x_t, score_target = self.forward_sample(dp1, dp2, t)
        score_pred = self.score_predict(x_t, t, dp1=dp1)
        sigma2_t = self._sigma2_at(t).reshape(-1, 1, 1) if x_t.dim() == 3 else self._sigma2_at(t).reshape(-1, 1)
        loss = F.mse_loss(score_pred, score_target, reduction="none")
        loss = (loss * sigma2_t).mean()
        return loss

    # ── Interpretability diagnostics ──────────────────────────────────────────

    @torch.no_grad()
    def baseline_loss(self, x_like: torch.Tensor) -> torch.Tensor:
        """
        The score-matching MSE achieved by the no-signal predictor s = 0.

        With s = (mu - x)/sigma^2 and x = mu + sigma*z, the sigma^2-weighted
        MSE for a zero score is sigma^2 * ||z/sigma||^2 = ||z||^2, so
        E[baseline] = mean over dimensions of E[z^2] = 1.

        This is the reference point that makes the raw loss interpretable:
        a loss of ~1.0 means the network is still predicting ~nothing.
        ``x_like`` supplies only the batch/device/dtype shape.
        """
        z = torch.randn_like(x_like)
        # sigma^2-weighted MSE with zero prediction == ||z||^2 elementwise mean.
        return (z ** 2).mean()

    @torch.no_grad()
    def signal_captured(
        self,
        dp1: torch.Tensor,
        dp2: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        num_eval: int = 16,
    ) -> Tuple[float, float, float]:
        """
        Compare the actual score-matching loss against the zero-score baseline.

        Returns (loss, baseline, signal) where
            signal = (baseline - loss) / baseline
        is the fraction of the score signal the network has captured. signal ~ 0
        means it predicts nothing; signal -> 1 means it matches the true score.

        ``num_eval`` uses a subset of the batch to keep this cheap; pass
        ``t`` to evaluate at specific noise levels.
        """
        loss = self.score_matching_loss(dp1[:num_eval], dp2[:num_eval], t=t)
        baseline = self.baseline_loss(dp1[:num_eval])
        signal = (baseline - loss) / baseline.clamp(min=1e-6)
        return loss.item(), baseline.item(), signal.item()

    @torch.no_grad()
    def reconstruction_error(
        self,
        dp1: torch.Tensor,
        dp2: torch.Tensor,
        steps: Optional[int] = None,
        num_eval: int = 16,
    ) -> torch.Tensor:
        """
        Mean ||sample(dp1) - dp2|| for a full reverse-SDE generation, evaluated
        on a small slice of ``dp1``/``dp2``.

        This is the direct question "is DP1 -> DP2 being learned?": the score
        network drives the reverse SDE to an output embedding; the error vs.
        the true DP2 drops as generation improves. Unlike recovering the target
        from a single noisy ``x_t`` (which amplifies noise at low t), the full
        reverse sample integrates the score over all steps and is robust.
        """
        dp1_e = dp1[:num_eval]
        dp2_e = dp2[:num_eval]
        out = self.sample(dp1_e, steps=steps)
        return torch.norm(out - dp2_e, dim=-1).mean()

    # ── Reverse SDE: generate DP2 from DP1 ────────────────────────────────────

    def _estimate_target(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dp1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recover the (unknown) output DP2 from the learned score.

        The score is s = (mu - x) / sigma^2 with mu = (1 - alpha) DP2 + alpha DP1,
        so DP2 = (x + sigma^2 s - alpha DP1) / (1 - alpha). This lets the
        reverse drift point along the relative displacement toward the output.
        """
        alpha_t = self._alpha_at(t).reshape(-1, 1, 1) if x.dim() == 3 else self._alpha_at(t).reshape(-1, 1)
        sigma2_t = self._sigma2_at(t).reshape_as(alpha_t)
        s = self.score_predict(x, t, dp1=dp1)
        mu = x + sigma2_t * s
        denom = (1 - alpha_t).clamp(min=1e-3)
        return (mu - alpha_t * dp1) / denom

    @torch.no_grad()
    def sample(
        self,
        dp1: torch.Tensor,
        steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Generate DP2 from DP1 by integrating the bridge FORWARD in time.

        The bridge is pinned at DP1 (t=0, sigma^2~0) and pulled toward DP2
        (t=1). Generation therefore integrates the forward SDE from t=0 to
        t=1, replacing the unknown DP2 in the drift with its score-based
        estimate at each step, and returns the final target estimate (not the
        noisy terminal state x, whose marginal variance sigma^2(1)~0.5 is
        large).

        (The previous version evaluated coefficients at t going 1 -> 0 while
        integrating the state forward, which started the trajectory at an
        off-distribution point — DP1 is not a sample of N(DP2, 0.5 I) — and
        made reconstruction_error worse than the identity baseline.)

        Args:
            dp1: (B, D) or (B, S, D) input.
            steps: number of integration steps (defaults to num_steps).
            return_trajectory: if True, return all intermediate states.

        Returns:
            (B, D) or (B, S, D) estimated DP2, or (steps+1, B, ...) if
            return_trajectory.
        """
        steps = steps or self.num_steps
        dt = 1.0 / steps
        x = dp1.clone()
        traj = [x]
        dp2_est = dp1.clone()
        for i in range(steps):
            t = torch.full((x.shape[0],), (i + 0.5) * dt, device=x.device)
            sigma2_t = self._sigma2_at(t).reshape(-1, 1, 1) if x.dim() == 3 else self._sigma2_at(t).reshape(-1, 1)
            beta_t = self._beta_at(t).reshape_as(sigma2_t)
            # Forward drift toward the score-recovered target.
            dp2_est = self._estimate_target(x, t, dp1)
            drift = beta_t * (dp2_est - x)
            noise = torch.randn_like(x)
            x = x + drift * dt + torch.sqrt(beta_t * dt + 1e-8) * noise
            if return_trajectory:
                traj.append(x.clone())
        if return_trajectory:
            return torch.stack(traj)
        return dp2_est