#!/usr/bin/env python3
"""
Visualize Diffusion Schrödinger Bridge (DSB) Semantic Basins, Sinks, and SDE Trajectories.

This script visualizes:
  1. Semantic Sinks (Attractors): Clean sentence embeddings forming energy minima.
  2. Basins of Attraction ("Blobs"): Clouds of corrupted/noisy variations orbiting each sink.
  3. The SDE Bridge Trajectory: Continuous path traced by bridge.sample() from DP1 -> DP2.
  4. Vector Drift Flow Field (Quiver): The learned restoring vector field pushing points into sinks.

Outputs:
  - Interactive HTML visualization (Plotly via CDN - open in browser)
  - High-res static PNG (Matplotlib)

Usage:
    python scripts/visualize_dsb_bridge.py \
        --checkpoint checkpoints_dsb_hybrid/best.pt \
        --out_html dsb_landscape.html \
        --out_png dsb_landscape.png
"""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from dllm.dsb import DiffSchrodingerBridge, MLPScoreNet, TransformerScoreNet
from dllm.dsb_hybrid import DSBHybrid, EditConditionedScoreNet, corrupt_fixed
from dllm.utils import resolve_device, set_seed


DEFAULT_SENTENCES = [
    # Concept 1: Geography (English + Hindi parallel)
    "The capital of France is Paris.",
    "फ्रांस की राजधानी पेरिस है।",
    # Concept 2: Animal / Nature
    "The quick brown fox jumps over the lazy dog.",
    "एक फुर्तीली लोमड़ी कुत्ते के ऊपर से कूदती है।",
    # Concept 3: Science / Technology
    "Artificial intelligence models learn patterns from data.",
    "मशीन लर्निंग मॉडल डेटा से सीखते हैं।",
    # Concept 4: Identity / Location
    "India is a vast and diverse country.",
    "भारत एक विशाल और विविधतापूर्ण देश है।",
]


class TextEmbedder(torch.nn.Module):
    def __init__(self, backbone="xlm-roberta-base", max_length=128):
        super().__init__()
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.model = AutoModelForMaskedLM.from_pretrained(backbone)
        self.encoder = getattr(self.model, self.model.base_model_prefix, self.model)
        self.lm_head = getattr(self.model, "lm_head", getattr(self.model, "cls", None))
        self.max_length = max_length
        self.dim = self.model.config.hidden_size
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def embed_pool(self, texts, device):
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.encoder(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return pooled  # (B, D)

    def embed_ids(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state  # (B, S, D)

    def decode_logits(self, hidden_states):
        """Decode Layer 12 contextual hidden states (..., D) -> (..., V) vocabulary logits."""
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        elif hasattr(self.model, "get_output_embeddings"):
            return self.model.get_output_embeddings()(hidden_states)
        else:
            raise AttributeError("Model has no lm_head")


def load_hybrid_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    mcfg = config["model"]
    
    embedder = TextEmbedder(backbone=mcfg.get("embedder", "xlm-roberta-base"),
                            max_length=mcfg.get("max_length", 128)).to(device)
    if "embedder" in ckpt:
        try:
            embedder.load_state_dict(ckpt["embedder"])
        except Exception:
            pass
    embedder.eval()

    cond_on = bool(config["dsb"].get("condition_on_dp1", False))
    cond_dim = embedder.dim if cond_on else 0

    score_type = mcfg.get("score_net", "mlp")
    if "hybrid" in ckpt:
        sd = ckpt["hybrid"]
        if any("bridge.score_net.encoder" in k for k in sd):
            score_type = "transformer"
        elif any("bridge.score_net.tag_emb" in k for k in sd):
            score_type = "edit_conditioned"

    if score_type == "transformer":
        score_net = TransformerScoreNet(
            dim=embedder.dim, hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"], time_embed_dim=mcfg["time_embed_dim"],
            cond_dim=cond_dim, num_heads=mcfg.get("num_heads", 8),
        )
    elif score_type == "edit_conditioned":
        score_net = EditConditionedScoreNet(
            dim=embedder.dim, num_tags=5, hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"], time_embed_dim=mcfg["time_embed_dim"],
            cond_dim=cond_dim,
        )
    else:
        score_net = MLPScoreNet(dim=embedder.dim, hidden_dim=mcfg["hidden_dim"],
                                num_layers=mcfg["num_layers"], time_embed_dim=mcfg["time_embed_dim"],
                                cond_dim=cond_dim)

    bridge = DiffSchrodingerBridge(
        dim=embedder.dim, score_net=score_net,
        beta_schedule=config["dsb"]["beta_schedule"],
        num_steps=config["dsb"]["num_steps"],
        beta_min=config["dsb"]["beta_min"], beta_max=config["dsb"]["beta_max"],
        condition_on_dp1=cond_on,
        sigma2_schedule=config["dsb"].get("sigma2_schedule", "bridge"),
        prediction_target=config["dsb"].get("prediction_target", "x0"),
    ).to(device)

    if "hybrid" in ckpt:
        embed_weight = embedder.encoder.get_input_embeddings().weight
        lm_head = getattr(embedder, "lm_head", None)
        hybrid = DSBHybrid(
            bridge=bridge, vocab_size=embedder.tokenizer.vocab_size,
            condition_heads=bool(mcfg.get("condition_heads", False)),
            time_embed_dim=mcfg.get("time_embed_dim", 128),
            embed_weight=embed_weight,
            tie_weights=mcfg.get("tie_weights", True),
            lm_head=lm_head,
        ).to(device)
        hybrid.load_state_dict(ckpt["hybrid"])
        hybrid.eval()
        return embedder, hybrid.bridge, hybrid
    else:
        bridge.score_net.load_state_dict(ckpt["score_net"])
        bridge.eval()
        return embedder, bridge, None


def generate_variations(clean_text, tokenizer, num_variations=15):
    """Generate noisy and corrupted variations around a clean text."""
    tokens = tokenizer.encode(clean_text, add_special_tokens=False)
    variations = []
    mask_id = tokenizer.mask_token_id or 250001
    noise_pool = list(range(100, min(10000, tokenizer.vocab_size)))
    
    for i in range(num_variations):
        mask_prob = 0.15 + 0.5 * (i / max(1, num_variations - 1))
        corr, _, _ = corrupt_fixed(
            tokens, mask_prob=mask_prob, mask_ratio=0.7,
            noise_pool=noise_pool, mask_id=mask_id
        )
        text_corr = tokenizer.decode(corr).strip()
        variations.append((text_corr, mask_prob))
    return variations


def perform_pca_fit(X, n_components=2):
    """Fit PCA using SVD in numpy."""
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:n_components]
    return mean, components


def project_pca(X, mean, components):
    return (X - mean) @ components.T


def inverse_project_pca(X_2d, mean, components):
    return X_2d @ components + mean


def build_interactive_html(
    sinks_2d,
    sink_labels,
    blobs_2d,
    blob_labels,
    blob_groups,
    traj_2d,
    quiver_grid,
    quiver_uv,
    out_path="dsb_landscape.html"
):
    """Generate a standalone HTML visualization powered by Plotly.js CDN."""
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
    ]

    data_traces = []

    # 1. Quiver flow field arrows
    if quiver_grid is not None and quiver_uv is not None:
        q_x, q_y = quiver_grid[:, 0], quiver_grid[:, 1]
        u, v = quiver_uv[:, 0], quiver_uv[:, 1]
        norm = np.sqrt(u**2 + v**2).clip(min=1e-5)
        scale = 0.35
        u_scaled, v_scaled = (u / norm) * scale, (v / norm) * scale
        
        arrow_x, arrow_y = [], []
        for gx, gy, gu, gv in zip(q_x, q_y, u_scaled, v_scaled):
            arrow_x.extend([gx, gx + gu, None])
            arrow_y.extend([gy, gy + gv, None])

        data_traces.append({
            "type": "scatter",
            "mode": "lines",
            "x": arrow_x,
            "y": arrow_y,
            "line": {"color": "rgba(180, 190, 205, 0.45)", "width": 1.2},
            "name": "Score Drift Field ∇ log p(x)",
            "hoverinfo": "skip"
        })

    # 2. Blobs / Basins of Attraction (Noisy clouds)
    unique_groups = sorted(list(set(blob_groups)))
    for g in unique_groups:
        idx = [i for i, grp in enumerate(blob_groups) if grp == g]
        color = colors[g % len(colors)]
        data_traces.append({
            "type": "scatter",
            "mode": "markers",
            "x": [float(blobs_2d[i, 0]) for i in idx],
            "y": [float(blobs_2d[i, 1]) for i in idx],
            "text": [blob_labels[i] for i in idx],
            "hoverinfo": "text",
            "marker": {
                "size": 7,
                "color": color,
                "opacity": 0.35,
                "symbol": "circle",
            },
            "name": f"Basin {g+1}: {sink_labels[g][:25]}...",
        })

    # 3. Clean Sinks (Attractors)
    data_traces.append({
        "type": "scatter",
        "mode": "markers+text",
        "x": [float(sinks_2d[i, 0]) for i in range(len(sinks_2d))],
        "y": [float(sinks_2d[i, 1]) for i in range(len(sinks_2d))],
        "text": [f"🌟 Sink {i+1}" for i in range(len(sinks_2d))],
        "textposition": "top center",
        "hovertext": [f"<b>Clean Sink {i+1}:</b><br>{lbl}" for i, lbl in enumerate(sink_labels)],
        "hoverinfo": "text",
        "marker": {
            "size": 16,
            "color": [colors[i % len(colors)] for i in range(len(sinks_2d))],
            "symbol": "star-diamond",
            "line": {"color": "#111", "width": 2}
        },
        "name": "Clean Sinks (Truths / Attractors)"
    })

    # 4. SDE Bridge Trajectory
    if traj_2d is not None and len(traj_2d) > 0:
        data_traces.append({
            "type": "scatter",
            "mode": "lines+markers",
            "x": [float(p[0]) for p in traj_2d],
            "y": [float(p[1]) for p in traj_2d],
            "line": {"color": "#e63946", "width": 3.5},
            "marker": {
                "size": [10 if i in (0, len(traj_2d)-1) else 4 for i in range(len(traj_2d))],
                "color": "#d90429",
            },
            "text": [f"SDE step {i}/{len(traj_2d)-1} (t={i/(len(traj_2d)-1):.2f})" for i in range(len(traj_2d))],
            "hoverinfo": "text",
            "name": "SDE Bridge Geodesic (DP1 → DP2)"
        })

    layout = {
        "title": {
            "text": "🌉 Diffusion Schrödinger Bridge: Semantic Sinks, Attractor Basins & SDE Transport",
            "font": {"size": 20, "color": "#1a202c"}
        },
        "plot_bgcolor": "#f8fafc",
        "paper_bgcolor": "#ffffff",
        "xaxis": {"title": "PCA Dimension 1", "gridcolor": "#e2e8f0", "zeroline": False},
        "yaxis": {"title": "PCA Dimension 2", "gridcolor": "#e2e8f0", "zeroline": False},
        "hovermode": "closest",
        "legend": {"orientation": "h", "y": -0.15, "x": 0.0},
        "width": 1100,
        "height": 780,
    }

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>DSB Semantic Basins and Sinks Visualization</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #f1f5f9;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .card {{
            background: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            max-width: 1140px;
            width: 100%;
        }}
        .desc {{
            color: #475569;
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 20px;
            background: #f8fafc;
            padding: 16px;
            border-left: 4px solid #3b82f6;
            border-radius: 4px;
        }}
        .desc b {{ color: #1e293b; }}
    </style>
</head>
<body>
    <div class="card">
        <h2>🌉 Diffusion Schrödinger Bridge (DSB) State-Space Manifold</h2>
        <div class="desc">
            <b>How to read this visualization:</b><br>
            • <b>🌟 Sinks (Stars)</b>: Clean statements/truths embedded into continuous representation space.<br>
            • <b>Clouds (Dots)</b>: Noisy & masked variations forming the <i>basin of attraction</i> around each clean sink.<br>
            • <b>Grey Arrows (Quiver)</b>: The learned score vector field ∇<sub>x</sub> log p(x) pointing downhill into the nearest sink.<br>
            • <b>Red Curve</b>: The continuous SDE path transporting a corrupted prompt (DP1) into the clean target basin (DP2).
        </div>
        <div id="plot"></div>
    </div>
    <script>
        var data = {json.dumps(data_traces)};
        var layout = {json.dumps(layout)};
        Plotly.newPlot('plot', data, layout, {{responsive: true}});
    </script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved interactive HTML visualization to {out_path}")


def build_static_png(
    sinks_2d,
    sink_labels,
    blobs_2d,
    blob_groups,
    traj_2d,
    quiver_grid,
    quiver_uv,
    out_path="dsb_landscape.png"
):
    """Generate high-resolution PNG using matplotlib."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8), dpi=200)
    ax = plt.gca()
    ax.set_facecolor("#f8fafc")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
              "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    # 1. Quiver field
    if quiver_grid is not None and quiver_uv is not None:
        q_x, q_y = quiver_grid[:, 0], quiver_grid[:, 1]
        u, v = quiver_uv[:, 0], quiver_uv[:, 1]
        norm = np.sqrt(u**2 + v**2).clip(min=1e-5)
        u_norm, v_norm = u / norm, v / norm
        ax.quiver(q_x, q_y, u_norm, v_norm, color="#cbd5e1", alpha=0.5,
                  width=0.002, scale=30, headwidth=4, headlength=4)

    # 2. Blobs
    unique_groups = sorted(list(set(blob_groups)))
    for g in unique_groups:
        idx = [i for i, grp in enumerate(blob_groups) if grp == g]
        color = colors[g % len(colors)]
        ax.scatter(blobs_2d[idx, 0], blobs_2d[idx, 1], c=color, alpha=0.3, s=30,
                   label=f"Basin {g+1}" if g < 4 else "")

    # 3. Sinks
    for i, (sx, sy) in enumerate(sinks_2d):
        color = colors[i % len(colors)]
        ax.scatter([sx], [sy], c=color, s=180, marker="*", edgecolor="#0f172a", linewidth=1.5, zorder=5)
        ax.text(sx, sy + 0.15, f"Sink {i+1}", fontsize=9, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cbd5e1", alpha=0.85))

    # 4. Trajectory
    if traj_2d is not None and len(traj_2d) > 0:
        ax.plot(traj_2d[:, 0], traj_2d[:, 1], color="#e63946", lw=2.5, zorder=6, label="SDE Bridge Trajectory (DP1 → DP2)")
        ax.scatter(traj_2d[0, 0], traj_2d[0, 1], c="#1d3557", s=80, marker="o", label="DP1 (Start)", zorder=7)
        ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], c="#e63946", s=100, marker="X", label="DP2 (Target)", zorder=7)

    ax.set_title("Diffusion Schrödinger Bridge: Semantic Sinks, Attractor Basins & SDE Transport",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("PCA Dimension 1", fontsize=11)
    ax.set_ylabel("PCA Dimension 2", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved static PNG visualization to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize DSB semantic basins, sinks, and trajectories")
    parser.add_argument("--checkpoint", default="checkpoints_dsb_hybrid/best.pt", help="Path to DSB checkpoint (.pt)")
    parser.add_argument("--sentences", nargs="+", default=None, help="List of clean sentences (sinks)")
    parser.add_argument("--num_variations", type=int, default=20, help="Number of corrupted variations per sink")
    parser.add_argument("--sde_steps", type=int, default=50, help="Number of SDE reverse integration steps")
    parser.add_argument("--out_html", default="dsb_landscape.html", help="Path to output HTML")
    parser.add_argument("--out_png", default="dsb_landscape.png", help="Path to output PNG")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(resolve_device() if args.device is None else args.device)
    print(f"Using device: {device}")

    sentences = args.sentences or DEFAULT_SENTENCES
    print(f"Visualizing {len(sentences)} clean semantic sinks...")

    # 1. Load model
    embedder, bridge, hybrid = load_hybrid_model(args.checkpoint, device)
    tokenizer = embedder.tokenizer

    # 2. Embed Sinks (Clean Sentences)
    sink_embs = embedder.embed_pool(sentences, device).cpu().numpy()  # (K, D)

    # 3. Generate and Embed Corrupted Variations (Basins / Blobs)
    all_blob_embs = []
    all_blob_labels = []
    all_blob_groups = []

    for k, s in enumerate(sentences):
        variations = generate_variations(s, tokenizer, num_variations=args.num_variations)
        var_texts = [v[0] for v in variations]
        if var_texts:
            v_embs = embedder.embed_pool(var_texts, device).cpu().numpy()
            all_blob_embs.append(v_embs)
            for txt, p in variations:
                all_blob_labels.append(f"<b>Noisy var (p={p:.2f}):</b><br>{txt}")
                all_blob_groups.append(k)

    blob_embs = np.vstack(all_blob_embs) if all_blob_embs else np.empty((0, embedder.dim))

    # 4. Compute SDE Diffusion Trajectory for Sentence 0
    # Choose first sentence corrupted as prompt DP1
    target_clean = sentences[0]
    prompt_corr, _, _ = corrupt_fixed(
        tokenizer.encode(target_clean, add_special_tokens=False),
        mask_prob=0.5, mask_ratio=0.8,
        mask_id=tokenizer.mask_token_id or 250001
    )
    prompt_text = tokenizer.decode(prompt_corr)
    print(f"Generating SDE trajectory from corrupted prompt: {prompt_text!r}")

    dp1_single = embedder.embed_pool([prompt_text], device)  # (1, D)
    traj_tensor = bridge.sample(dp1_single, steps=args.sde_steps, return_trajectory=True)  # (Steps+1, 1, D)
    traj_embs = traj_tensor.squeeze(1).cpu().numpy()  # (Steps+1, D)

    # 5. Fit 2D PCA on all representations
    combined_data = np.vstack([sink_embs, blob_embs, traj_embs])
    pca_mean, pca_components = perform_pca_fit(combined_data, n_components=2)

    sinks_2d = project_pca(sink_embs, pca_mean, pca_components)
    blobs_2d = project_pca(blob_embs, pca_mean, pca_components)
    traj_2d = project_pca(traj_embs, pca_mean, pca_components)

    # 6. Compute Quiver Vector Field Grid
    x_min, x_max = combined_data_2d = project_pca(combined_data, pca_mean, pca_components)[:, 0].min(), project_pca(combined_data, pca_mean, pca_components)[:, 0].max()
    y_min, y_max = project_pca(combined_data, pca_mean, pca_components)[:, 1].min(), project_pca(combined_data, pca_mean, pca_components)[:, 1].max()
    pad_x, pad_y = 0.2 * (x_max - x_min), 0.2 * (y_max - y_min)

    gx = np.linspace(x_min - pad_x, x_max + pad_x, 15)
    gy = np.linspace(y_min - pad_y, y_max + pad_y, 15)
    g_xx, g_yy = np.meshgrid(gx, gy)
    grid_2d = np.stack([g_xx.ravel(), g_yy.ravel()], axis=-1)  # (225, 2)

    grid_hd = inverse_project_pca(grid_2d, pca_mean, pca_components)  # (225, D)
    grid_tensor = torch.tensor(grid_hd, dtype=torch.float32, device=device)
    t_mid = torch.full((len(grid_tensor),), 0.5, device=device)
    dp1_rep = dp1_single.repeat(len(grid_tensor), 1)

    with torch.no_grad():
        u_pred = bridge.score_predict(grid_tensor, t_mid, dp1=dp1_rep)  # (225, D)
        drift_target = bridge._estimate_target(grid_tensor, t_mid, dp1_rep)
        drift_vec = (drift_target - grid_tensor).cpu().numpy()  # (225, D)

    quiver_uv = (drift_vec @ pca_components.T)  # (225, 2)

    # 7. Render Visualizations
    build_interactive_html(
        sinks_2d, sentences, blobs_2d, all_blob_labels, all_blob_groups,
        traj_2d, grid_2d, quiver_uv, out_path=args.out_html
    )
    build_static_png(
        sinks_2d, sentences, blobs_2d, all_blob_groups,
        traj_2d, grid_2d, quiver_uv, out_path=args.out_png
    )
    print("\nVisualization generation complete!")


if __name__ == "__main__":
    main()
