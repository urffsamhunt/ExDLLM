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


# ── Score network ─────────────────────────────────────────────────────────

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


class TransformerScoreNet(nn.Module):
    """
    A small transformer score network: self-attention across positions.

    Unlike ``MLPScoreNet`` (a per-position MLP), positions exchange information
    at intermediate SDE states, so denoising decisions can use the current
    state of OTHER positions and not just the context frozen into the encoder
    embeddings. This attacks the signal ceiling of the per-position MLP
    (~65% observed vs ~90% achievable). A (B, D) input is treated as a
    length-1 sequence, so pooled (plain-DSB) inputs also work.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        time_embed_dim: int = 128,
        cond_dim: int = 0,
        num_heads: int = 8,
        dropout: float = 0.0,
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
        self.in_proj = nn.Linear(dim + cond_dim + time_embed_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, S, D) or (B, D); t: (B,); cond: same shape as x (e.g. DP1)
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)                           # (B, 1, D)
            if cond is not None:
                cond = cond.unsqueeze(1)
        h_in = x
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("score net built with cond_dim > 0 requires cond")
            h_in = torch.cat([cond, h_in], dim=-1)
        t_emb = self.time_mlp(t.reshape(-1, 1))          # (B, T)
        t_emb = t_emb.unsqueeze(1).expand(-1, h_in.shape[1], -1)
        h = self.in_proj(torch.cat([h_in, t_emb], dim=-1))
        pad_mask = (attention_mask == 0) if (attention_mask is not None and not squeeze) else None
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        out = self.out_proj(h)
        return out.squeeze(1) if squeeze else out


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
        sigma2_schedule: str = "ou",
        prediction_target: str = "x0",   # "x0" (direct DP2 prediction) or "u" (score matching)
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
        if prediction_target not in ("u", "x0"):
            raise ValueError(f"Unknown prediction_target: {prediction_target}")
        self.prediction_target = prediction_target

        if beta_schedule == "linear":
            betas = linear_beta_schedule(num_steps, beta_min, beta_max)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(num_steps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        self.register_buffer("betas", betas)

        # B_t = int_0^t beta_s ds (Euler discretization: cumulative sum)
        self.register_buffer("B", torch.cumsum(betas, dim=0))
        # alpha_t = exp(-B_t)
        self.register_buffer("alpha", torch.exp(-self.B))
        if sigma2_schedule not in ("ou", "bridge"):
            raise ValueError(f"Unknown sigma2_schedule: {sigma2_schedule}")
        self.sigma2_schedule = sigma2_schedule
        if sigma2_schedule == "bridge":
            # True two-ended bridge: variance vanishes at BOTH endpoints
            # (alpha=1 at t=0 pins DP1, alpha=0 at t=1 pins DP2), peak 0.5 at
            # alpha=0.5. The OU schedule instead saturates at sigma2(1)=0.5,
            # leaving ~sqrt(0.5*D) L2 of terminal noise that the score net
            # must cancel through dp2_est — the main cause of recon_err
            # plateauing above the identity baseline.
            sigma2 = 2.0 * self.alpha * (1.0 - self.alpha)
        else:
            sigma2 = (1.0 - torch.exp(-2.0 * self.B)) / 2.0
        self.register_buffer("sigma2", sigma2)

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
                      attention_mask: Optional[torch.Tensor] = None,
                      **kwargs) -> torch.Tensor:
        """Evaluate the score net, optionally conditioning on the source DP1."""
        cond = dp1 if self.condition_on_dp1 else None
        if isinstance(self.score_net, TransformerScoreNet):
            return self.score_net(x, t, cond=cond, attention_mask=attention_mask, **kwargs)
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
            target: training target — either u = mu_t - x_t (u-parametrization)
                    or dp2 directly (x0-prediction), depending on prediction_target.
        """
        t_b = t.reshape(-1, 1, 1) if dp1.dim() == 3 else t.reshape(-1, 1)
        alpha_t = self._alpha_at(t).reshape_as(t_b)
        sigma2_t = self._sigma2_at(t).reshape_as(t_b)

        mu = (1 - alpha_t) * dp2 + alpha_t * dp1  # linear bridge mean
        sigma_t = torch.sqrt(sigma2_t + 1e-8)
        z = torch.randn_like(dp1)
        x_t = mu + sigma_t * z

        if self.prediction_target == "x0":
            # x0-prediction: the network directly predicts the clean target DP2.
            # No algebraic inversion needed. No (1-alpha_t) singularity.
            return x_t, dp2
        else:
            # u-parametrization: predict u = mu - x_t (== -sigma_t * z) instead of
            # the raw score s = u / sigma2_t. The score spans ~4 orders of magnitude
            # across t (sigma2: 1e-4 -> 0.5), which a small MLP cannot represent —
            # the chronic cause of low "signal captured". Since
            # sigma2 * ||s_pred - s*||^2 == ||u_pred - u*||^2, the u-parametrization
            # is the SAME objective without the dynamic-range problem.
            u = mu - x_t
            return x_t, u

    # ── Training loss: denoising score matching ───────────────────────────────

    def score_matching_loss(
        self,
        dp1: torch.Tensor,
        dp2: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Denoising score-matching loss for the bridge.

        In u-parametrization (prediction_target='u'):
            L = E_t || u_theta(x_t, t, DP1) - (mu_t - x_t) ||^2

        In x0-prediction (prediction_target='x0'):
            L = E_t || x0_theta(x_t, t, DP1) - DP2 ||^2

        Args:
            dp1: (B, D) or (B, S, D) input.
            dp2: (B, D) or (B, S, D) output.
            t:   optional (B,) noise levels; sampled uniformly if None.
            attention_mask: optional (B, S) padding mask (1=real, 0=pad).

        Returns:
            scalar loss.
        """
        B = dp1.shape[0]
        if t is None:
            t = torch.rand(B, device=dp1.device)
        x_t, target = self.forward_sample(dp1, dp2, t)
        pred = self.score_predict(x_t, t, dp1=dp1, attention_mask=attention_mask)
        if attention_mask is not None and pred.dim() == 3:
            mask = attention_mask.unsqueeze(-1).float()
            loss = ((pred - target) ** 2 * mask).sum() / (mask.sum() * pred.shape[-1]).clamp(min=1.0)
        else:
            loss = F.mse_loss(pred, target)
        return loss

    # ── Interpretability diagnostics ──────────────────────────────────────────

    @torch.no_grad()
    def baseline_loss(self, x_like: torch.Tensor) -> torch.Tensor:
        """
        The MSE achieved by the zero predictor in the u-parametrization.

        u* = mu - x_t = -sigma_t * z, so the per-dim expected MSE for a zero
        prediction is E_t[sigma2_t]. This is the reference point that makes
        the raw loss interpretable: loss ~ baseline means the network is
        predicting ~nothing; loss -> 0 means it matches the target.
        """
        return self.sigma2.mean()

    @torch.no_grad()
    def signal_captured(
        self,
        dp1: torch.Tensor,
        dp2: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        num_eval: int = 16,
        num_t: int = 17,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float, float]:
        """
        Compare the actual score-matching loss against the zero-score baseline.

        Returns (loss, baseline, signal) where
            signal = (baseline - loss) / baseline
        is the fraction of the score signal the network has captured. signal ~ 0
        means it predicts nothing; signal -> 1 means it matches the true score.
        """
        if t is None:
            n = min(num_eval, dp1.shape[0])
            grid = torch.linspace(0.03, 0.97, num_t, device=dp1.device)
            t = grid.repeat(n)
            dp1_e = dp1[:n].repeat_interleave(num_t, dim=0)
            dp2_e = dp2[:n].repeat_interleave(num_t, dim=0)
            attn_e = attention_mask[:n].repeat_interleave(num_t, dim=0) if attention_mask is not None else None
        else:
            dp1_e = dp1[:num_eval]
            dp2_e = dp2[:num_eval]
            attn_e = attention_mask[:num_eval] if attention_mask is not None else None
        loss = self.score_matching_loss(dp1_e, dp2_e, t=t, attention_mask=attn_e)
        baseline = self._sigma2_at(t).mean()
        signal = (baseline - loss) / baseline.clamp(min=1e-6)
        return loss.item(), baseline.item(), signal.item()

    @torch.no_grad()
    def reconstruction_error(
        self,
        dp1: torch.Tensor,
        dp2: torch.Tensor,
        steps: Optional[int] = None,
        num_eval: int = 16,
        attention_mask: Optional[torch.Tensor] = None,
        return_clean: bool = True,
    ) -> torch.Tensor:
        """
        Mean ||sample(dp1) - dp2|| for a full reverse-SDE generation, evaluated
        on a small slice of ``dp1``/``dp2``.
        """
        dp1_e = dp1[:num_eval]
        dp2_e = dp2[:num_eval]
        attn_e = attention_mask[:num_eval] if attention_mask is not None else None
        out = self.sample(dp1_e, steps=steps, attention_mask=attn_e, return_clean=return_clean)
        return torch.norm(out - dp2_e, dim=-1).mean()

    # ── Reverse SDE: generate DP2 from DP1 ────────────────────────────────────

    def _estimate_target(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dp1: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Recover the (unknown) output DP2 from the learned score.

        In x0-prediction mode, the score net directly outputs DP2 — no
        algebraic inversion or damping needed.

        In u-parametrization, the score net predicts u = mu - x_t, so
        mu = x + u, and DP2 = (mu - alpha DP1) / (1 - alpha). This lets the
        reverse drift point along the relative displacement toward the output.
        """
        if self.prediction_target == "x0":
            # Direct prediction — the score net IS the target estimator.
            return self.score_predict(x, t, dp1=dp1, attention_mask=attention_mask)

        # u-parametrization path
        alpha_t = self._alpha_at(t).reshape(-1, 1, 1) if x.dim() == 3 else self._alpha_at(t).reshape(-1, 1)
        u = self.score_predict(x, t, dp1=dp1, attention_mask=attention_mask)
        mu = x + u
        denom = (1.0 - alpha_t).clamp(min=1e-4)
        est = (mu - alpha_t * dp1) / denom
        # Damping: only blend toward dp1 at the extreme initial boundary t ~ 0
        # (1 - alpha_t < 0.1) to prevent Euler division spikes, while allowing
        # est to reach DP2 cleanly across the rest of the bridge.
        blend = torch.clamp((1.0 - alpha_t) / 0.1, 0.0, 1.0)
        return blend * est + (1.0 - blend) * dp1

    @torch.no_grad()
    def sample(
        self,
        dp1: torch.Tensor,
        steps: Optional[int] = None,
        return_trajectory: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        return_clean: bool = True,
    ) -> torch.Tensor:
        """
        Generate DP2 from DP1 by stepping the bridge FORWARD in time.

        Uses the exact closed-form Markov transition step:
            x_{k+1} = (1 - alpha_{k->k+1}) * dp2_est + alpha_{k->k+1} * x_k + noise
        where:
            alpha_{k->k+1} = alpha_{k+1} / alpha_k
            sigma^2_{k->k+1} = max(0, sigma^2_{k+1} - (alpha_{k->k+1})^2 * sigma^2_k)

        This provides exact, stable forward integration from DP1 (t=0) to DP2 (t=1)
        without Euler discretization under-integration or noise explosion.
        """
        steps = steps or self.num_steps
        t_grid = torch.linspace(0.0, 1.0, steps + 1, device=dp1.device)
        x = dp1.clone()
        traj = [x]
        dp2_est = dp1.clone()

        is_3d = (dp1.dim() == 3)
        B = dp1.shape[0]

        for k in range(steps):
            t_k = t_grid[k]
            t_next = t_grid[k + 1]

            a_k = self._alpha_at(t_k)
            a_next = self._alpha_at(t_next)
            s2_k = self._sigma2_at(t_k)
            s2_next = self._sigma2_at(t_next)

            a_step = (a_next / (a_k + 1e-8)).clamp(0.0, 1.0)
            var_step = (s2_next - (a_step ** 2) * s2_k).clamp(min=0.0)

            # Broadcast shapes for 2D vs 3D tensors
            a_step_b = a_step.view(1, 1, 1) if is_3d else a_step.view(1, 1)
            var_step_b = var_step.view(1, 1, 1) if is_3d else var_step.view(1, 1)

            t_vec = torch.full((B,), t_k.item(), device=dp1.device)
            dp2_est = self._estimate_target(x, t_vec, dp1, attention_mask=attention_mask)

            x_mean = (1.0 - a_step_b) * dp2_est + a_step_b * x
            noise = torch.randn_like(x)
            x = x_mean + torch.sqrt(var_step_b + 1e-8) * noise

            if return_trajectory:
                traj.append(x.clone())

        if return_trajectory:
            return torch.stack(traj)
        # Return clean target estimate at t=1 (eliminating terminal residual noise from x_49)
        return dp2_est if return_clean else x