#!/usr/bin/env python3
"""
Visualize Diffusion Schrödinger Bridge (DSB) Semantic Basins, Sinks, SDE Trajectories,
and Iterative Discrete Edit Refinement Hops.

This script visualizes the full Two-Phase DSB Hybrid architecture:
  Phase 1: Continuous SDE Bridge Trajectory
    - Continuous geodesic flow in representation space from corrupted DP1 -> clean attractor basin DP2.
  Phase 2: Discrete Edit Refinement Hops
    - Discrete canvas token updates (KEEP / DELETE / REPLACE / INSERT) across iterations 0 -> K.
    - Each discrete iteration re-embeds the canvas, jumping closer to the true semantic sink.
  Plus:
    - Semantic Sinks (Attractors): Clean sentence embeddings forming energy minima.
    - Basins of Attraction ("Blobs"): Clouds of corrupted variations orbiting each sink.
    - Vector Drift Flow Field (Quiver): Learned restoring vector field ∇_x log p(x).
    - Interactive Discrete Edit Timeline Inspector (in HTML).

Outputs:
  - Interactive HTML visualization (Plotly via CDN - open in browser)
  - High-res static PNG (Matplotlib)

Usage:
    python scripts/visualize_dsb_bridge.py \
        --checkpoint checkpoints_dsb_hybrid/best.pt \
        --prompt "The capital of France is <mask_id>." \
        --target "The capital of France is Paris." \
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
from dllm.dsb_hybrid import (
    DSBHybrid,
    EditConditionedScoreNet,
    KEEP,
    DELETE,
    REPLACE,
    INSERT,
    EXPAND,
    TAG_NAMES,
    corrupt_fixed,
)
from dllm.utils import resolve_device, set_seed


DEFAULT_SENTENCES = [
    # Concept 1: Geography / Factual Capitals (English)
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    # Concept 2: Geography (Hindi parallel)
    "फ्रांस की राजधानी पेरिस है।",
    "भारत एक विशाल और विविधतापूर्ण देश है।",
    # Concept 3: Animal / Nature
    "The quick brown fox jumps over the lazy dog.",
    "एक फुर्तीली लोमड़ी कुत्ते के ऊपर से कूदती है।",
    # Concept 4: Science / Technology
    "Artificial intelligence models learn patterns from data.",
    "मशीन लर्निंग मॉडल डेटा से सीखते हैं।",
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
    config = ckpt.get("config", {})
    mcfg = config.get("model", {})
    dsb_cfg = config.get("dsb", {})
    sd = ckpt.get("hybrid", ckpt.get("model_state_dict", {}))

    embedder = TextEmbedder(backbone=mcfg.get("embedder", "xlm-roberta-base"),
                            max_length=mcfg.get("max_length", 128)).to(device)
    if "embedder" in ckpt:
        try:
            embedder.load_state_dict(ckpt["embedder"])
        except Exception:
            pass
    embedder.eval()

    cond_on = bool(dsb_cfg.get("condition_on_dp1", False))
    cond_dim = embedder.dim if cond_on else 0

    score_type = mcfg.get("score_net", "mlp")
    if any("bridge.score_net.encoder" in k for k in sd):
        score_type = "transformer"
    elif any("bridge.score_net.tag_emb" in k for k in sd):
        score_type = "edit_conditioned"

    gated_drift = bool(dsb_cfg.get("gated_drift", False) or mcfg.get("gated_drift", False) or any("bridge.score_net.router" in k for k in sd))

    if score_type == "transformer":
        score_net = TransformerScoreNet(
            dim=embedder.dim, hidden_dim=mcfg.get("hidden_dim", 512),
            num_layers=mcfg.get("num_layers", 3), time_embed_dim=mcfg.get("time_embed_dim", 128),
            cond_dim=cond_dim, num_heads=mcfg.get("num_heads", 8),
            gated_drift=gated_drift,
        )
    elif score_type == "edit_conditioned":
        score_net = EditConditionedScoreNet(
            dim=embedder.dim, num_tags=5, hidden_dim=mcfg.get("hidden_dim", 512),
            num_layers=mcfg.get("num_layers", 3), time_embed_dim=mcfg.get("time_embed_dim", 128),
            cond_dim=cond_dim,
        )
    else:
        score_net = MLPScoreNet(dim=embedder.dim, hidden_dim=mcfg.get("hidden_dim", 512),
                                num_layers=mcfg.get("num_layers", 3), time_embed_dim=mcfg.get("time_embed_dim", 128),
                                cond_dim=cond_dim)

    bridge = DiffSchrodingerBridge(
        dim=embedder.dim, score_net=score_net,
        beta_schedule=dsb_cfg.get("beta_schedule", "linear"),
        num_steps=dsb_cfg.get("num_steps", 1000),
        beta_min=dsb_cfg.get("beta_min", 0.0001),
        beta_max=dsb_cfg.get("beta_max", 0.02),
        condition_on_dp1=cond_on,
        sigma2_schedule=dsb_cfg.get("sigma2_schedule", "bridge"),
        prediction_target=dsb_cfg.get("prediction_target", "x0"),
    ).to(device)

    if "hybrid" in ckpt or any("tagger" in k for k in sd):
        embed_weight = embedder.encoder.get_input_embeddings().weight
        lm_head = getattr(embedder, "lm_head", None)
        subspace_enabled = bool(mcfg.get("subspace_factorization", False) or any("generator.lex_proj" in k for k in sd))
        macro_dim = int(mcfg.get("macro_dim", 512))
        lexical_dim = int(mcfg.get("lexical_dim", 256))
        lex_weight = float(mcfg.get("lexical_loss_weight", 1.5))
        ang_margin = float(mcfg.get("angular_margin", 0.05))
        m_scale = float(mcfg.get("margin_scale", 64.0))

        op_embed_dim = int(mcfg.get("op_embed_dim", 0))
        if any("generator.op_emb.weight" in k for k in sd):
            op_embed_dim = sd["generator.op_emb.weight"].shape[1]

        hybrid = DSBHybrid(
            bridge=bridge, vocab_size=embedder.tokenizer.vocab_size,
            condition_heads=bool(mcfg.get("condition_heads", False)),
            time_embed_dim=mcfg.get("time_embed_dim", 128),
            embed_weight=embed_weight,
            tie_weights=mcfg.get("tie_weights", True),
            lm_head=lm_head,
            subspace_factorization=subspace_enabled,
            macro_dim=macro_dim,
            lexical_dim=lexical_dim,
            lexical_loss_weight=lex_weight,
            angular_margin=ang_margin,
            margin_scale=m_scale,
            op_embed_dim=op_embed_dim,
        ).to(device)

        if mcfg.get("tie_weights", True) and "generator.net.3.weight" in sd:
            ckpt_tie = mcfg.get("tie_weights", False)
            if not ckpt_tie:
                sd = {k: v for k, v in sd.items() if k != "generator.net.3.weight"}
        hybrid.load_state_dict(sd, strict=False)
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
    traj_sde_2d,
    traj_discrete_2d,
    quiver_grid,
    quiver_uv,
    discrete_steps_data,
    out_path="dsb_landscape.html",
):
    """Generate a standalone HTML visualization powered by Plotly.js CDN."""
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#17becf"
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
            "line": {"color": "rgba(180, 190, 205, 0.40)", "width": 1.2},
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
                "opacity": 0.32,
                "symbol": "circle",
            },
            "name": f"Basin {g+1}: {sink_labels[g][:24]}...",
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
            "line": {"color": "#0f172a", "width": 2}
        },
        "name": "Clean Sinks (Truths / Attractors)"
    })

    # 4. Phase 1: Continuous SDE Bridge Trajectory
    if traj_sde_2d is not None and len(traj_sde_2d) > 0:
        data_traces.append({
            "type": "scatter",
            "mode": "lines+markers",
            "x": [float(p[0]) for p in traj_sde_2d],
            "y": [float(p[1]) for p in traj_sde_2d],
            "line": {"color": "#f97316", "width": 3, "dash": "dash"},
            "marker": {
                "size": [10 if i in (0, len(traj_sde_2d)-1) else 4 for i in range(len(traj_sde_2d))],
                "color": "#ea580c",
            },
            "text": [f"Phase 1 SDE Step {i}/{len(traj_sde_2d)-1} (t={i/(len(traj_sde_2d)-1):.2f})" for i in range(len(traj_sde_2d))],
            "hoverinfo": "text",
            "name": "Phase 1: Continuous SDE Flow (t: 0 → 1)"
        })

    # 5. Phase 2: Discrete Edit Refinement Hops
    if traj_discrete_2d is not None and len(traj_discrete_2d) > 0:
        step_labels = []
        for i, st in enumerate(discrete_steps_data):
            step_labels.append(f"<b>Phase 2: Discrete Hop {i}</b><br>Canvas: \"{st.get('text_after', '')}\"<br>Changed: {st.get('changed', False)}")

        data_traces.append({
            "type": "scatter",
            "mode": "lines+markers+text",
            "x": [float(p[0]) for p in traj_discrete_2d],
            "y": [float(p[1]) for p in traj_discrete_2d],
            "text": [f"H{i}" for i in range(len(traj_discrete_2d))],
            "textposition": "bottom right",
            "textfont": {"size": 11, "color": "#7c3aed"},
            "line": {"color": "#7c3aed", "width": 4},
            "marker": {
                "size": 12,
                "color": "#6d28d9",
                "symbol": "diamond",
                "line": {"color": "#ffffff", "width": 2}
            },
            "hovertext": step_labels,
            "hoverinfo": "text",
            "name": "Phase 2: Discrete Edit Hops (Iter 0 → K)"
        })

    layout = {
        "title": {
            "text": "🌉 DSB Hybrid: Continuous SDE Bridge & Discrete Edit Refinement Manifold",
            "font": {"size": 20, "color": "#0f172a"}
        },
        "plot_bgcolor": "#f8fafc",
        "paper_bgcolor": "#ffffff",
        "xaxis": {"title": "PCA Dimension 1", "gridcolor": "#e2e8f0", "zeroline": False},
        "yaxis": {"title": "PCA Dimension 2", "gridcolor": "#e2e8f0", "zeroline": False},
        "hovermode": "closest",
        "legend": {"orientation": "h", "y": -0.15, "x": 0.0},
        "width": 1100,
        "height": 760,
    }

    # Format HTML table rows for discrete steps
    table_rows = []
    tag_badge_class = {
        KEEP: "badge-keep",
        DELETE: "badge-del",
        REPLACE: "badge-rep",
        INSERT: "badge-ins",
        EXPAND: "badge-exp"
    }

    for step in discrete_steps_data:
        it = step.get("iteration", 0)
        txt = step.get("text_after", "")
        tok_details = step.get("token_details", [])

        tok_html_list = []
        for td in tok_details:
            tg = td.get("tag", KEEP)
            t_str = td.get("token_str", "").replace("<", "&lt;").replace(">", "&gt;")
            badge = tag_badge_class.get(tg, "badge-keep")
            tag_name = td.get("tag_name", "KEEP")
            act = td.get("action_str", "")
            act_disp = f" → {act}" if act else ""
            conf = td.get("conf", 1.0)
            tok_html_list.append(f'<span class="tok-chip {badge}" title="{tag_name} (conf={conf:.2f}){act_disp}">{t_str}</span>')

        tok_chips = " ".join(tok_html_list)
        cos_disp = f"{step.get('cos_sim', 0.0):.4f}" if "cos_sim" in step else "—"
        status_disp = '<span class="status-mod">MODIFIED</span>' if step.get("changed", False) else '<span class="status-conv">CONVERGED</span>'

        table_rows.append(f"""
        <tr>
            <td style="font-weight: bold; text-align: center;">Iter {it}</td>
            <td><div class="chips-container">{tok_chips}</div></td>
            <td><code>"{txt}"</code></td>
            <td style="text-align: center;">{cos_disp}</td>
            <td style="text-align: center;">{status_disp}</td>
        </tr>
        """)

    table_body_html = "\n".join(table_rows)

    def _json_default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    json_data = json.dumps(data_traces, default=_json_default)
    json_layout = json.dumps(layout, default=_json_default)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>DSB Hybrid Manifold & Edit Inspector</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #f1f5f9;
            margin: 0;
            padding: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .card {{
            background: white;
            padding: 28px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            max-width: 1140px;
            width: 100%;
            margin-bottom: 24px;
        }}
        .desc {{
            color: #475569;
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 20px;
            background: #f8fafc;
            padding: 16px 20px;
            border-left: 4px solid #7c3aed;
            border-radius: 4px;
        }}
        .desc b {{ color: #0f172a; }}
        .badge-keep {{ background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }}
        .badge-rep  {{ background: #dbeafe; color: #1e40af; border: 1px solid #bfdbfe; }}
        .badge-ins  {{ background: #f3e8ff; color: #6b21a8; border: 1px solid #e9d5ff; }}
        .badge-del  {{ background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }}
        .badge-exp  {{ background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }}
        .tok-chip {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
            font-family: "JetBrains Mono", monospace;
            margin: 2px;
        }}
        .chips-container {{
            max-width: 380px;
            display: flex;
            flex-wrap: wrap;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-top: 12px;
        }}
        th, td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e2e8f0;
            text-align: left;
        }}
        th {{
            background: #f8fafc;
            color: #334155;
            font-weight: 600;
        }}
        .status-mod {{ color: #d97706; font-weight: 600; }}
        .status-conv {{ color: #16a34a; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="card">
        <h2>🌉 Diffusion Schrödinger Bridge (DSB) Hybrid Visualizer</h2>
        <div class="desc">
            <b>Two-Phase Hybrid Architecture in Action:</b><br>
            • <b>🌟 Sinks (Stars)</b>: Clean factual target sentences forming semantic energy minima.<br>
            • <b>Clouds (Dots)</b>: Noisy & masked variations forming the <i>basin of attraction</i> around each clean sink.<br>
            • <b>Grey Arrows (Quiver)</b>: Learned restoring score vector field ∇<sub>x</sub> log p(x).<br>
            • <b>🟠 Phase 1 (Orange Dashed)</b>: Continuous SDE Bridge Geodesic transported by <code>bridge.sample()</code> ($t=0 \to 1$).<br>
            • <b>🟣 Phase 2 (Purple Line)</b>: Discrete Refinement Hops where the discrete edit grammar physically mutates the canvas.
        </div>
        <div id="plot"></div>
    </div>

    <div class="card">
        <h3>🔍 Phase 2: Discrete Edit Refinement Inspector</h3>
        <p style="color: #64748b; font-size: 13px;">Evolution of token canvas, predicted tags, and semantic cosine similarity to target sink across discrete refinement iterations:</p>
        <table>
            <thead>
                <tr>
                    <th style="width: 70px; text-align: center;">Round</th>
                    <th>Canvas Token Tags</th>
                    <th>Decoded Canvas Text</th>
                    <th style="width: 130px; text-align: center;">Cos Sim to Target</th>
                    <th style="width: 110px; text-align: center;">Status</th>
                </tr>
            </thead>
            <tbody>
                {table_body_html}
            </tbody>
        </table>
    </div>

    <script>
        var data = {json_data};
        var layout = {json_layout};
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
    traj_sde_2d,
    traj_discrete_2d,
    quiver_grid,
    quiver_uv,
    out_path="dsb_landscape.png",
):
    """Generate high-resolution PNG using matplotlib."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(13, 9), dpi=200)
    ax = plt.gca()
    ax.set_facecolor("#f8fafc")

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#17becf"
    ]

    # 1. Quiver field
    if quiver_grid is not None and quiver_uv is not None:
        q_x, q_y = quiver_grid[:, 0], quiver_grid[:, 1]
        u, v = quiver_uv[:, 0], quiver_uv[:, 1]
        norm = np.sqrt(u**2 + v**2).clip(min=1e-5)
        u_norm, v_norm = u / norm, v / norm
        ax.quiver(q_x, q_y, u_norm, v_norm, color="#cbd5e1", alpha=0.45,
                  width=0.002, scale=30, headwidth=4, headlength=4)

    # 2. Blobs
    unique_groups = sorted(list(set(blob_groups)))
    for g in unique_groups:
        idx = [i for i, grp in enumerate(blob_groups) if grp == g]
        color = colors[g % len(colors)]
        ax.scatter(blobs_2d[idx, 0], blobs_2d[idx, 1], c=color, alpha=0.25, s=28,
                   label=f"Basin {g+1}" if g < 4 else "")

    # 3. Sinks
    for i, (sx, sy) in enumerate(sinks_2d):
        color = colors[i % len(colors)]
        ax.scatter([sx], [sy], c=color, s=190, marker="*", edgecolor="#0f172a", linewidth=1.5, zorder=6)
        ax.text(sx, sy + 0.14, f"Sink {i+1}", fontsize=8.5, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cbd5e1", alpha=0.85), zorder=7)

    # 4. Phase 1 Continuous SDE Trajectory
    if traj_sde_2d is not None and len(traj_sde_2d) > 0:
        ax.plot(traj_sde_2d[:, 0], traj_sde_2d[:, 1], color="#ea580c", lw=2.2, linestyle="--",
                zorder=7, label="Phase 1: Continuous SDE Bridge (t: 0 → 1)")
        ax.scatter(traj_sde_2d[0, 0], traj_sde_2d[0, 1], c="#ea580c", s=70, marker="o", zorder=8)
        ax.scatter(traj_sde_2d[-1, 0], traj_sde_2d[-1, 1], c="#ea580c", s=90, marker="s", zorder=8)

    # 5. Phase 2 Discrete Refinement Hops
    if traj_discrete_2d is not None and len(traj_discrete_2d) > 0:
        ax.plot(traj_discrete_2d[:, 0], traj_discrete_2d[:, 1], color="#7c3aed", lw=3.0,
                zorder=9, label="Phase 2: Discrete Edit Refinement Hops")
        for i, (dx, dy) in enumerate(traj_discrete_2d):
            ax.scatter([dx], [dy], c="#6d28d9", s=90, marker="D", edgecolor="#ffffff", linewidth=1.5, zorder=10)
            ax.text(dx + 0.08, dy - 0.08, f"H{i}", fontsize=8, fontweight="bold", color="#5b21b6", zorder=11)

    ax.set_title("Diffusion Schrödinger Bridge: Continuous SDE Trajectory & Discrete Refinement Hops",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("PCA Dimension 1", fontsize=11)
    ax.set_ylabel("PCA Dimension 2", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved static PNG visualization to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize DSB semantic basins, sinks, SDE, and discrete refinement")
    parser.add_argument("--checkpoint", default="checkpoints_dsb_hybrid/best.pt", help="Path to DSB checkpoint (.pt)")
    parser.add_argument("--sentences", nargs="+", default=None, help="List of clean sentences (sinks)")
    parser.add_argument("--prompt", default="The capital of France is <mask_id>.", help="Input prompt to run bridge and refinement on")
    parser.add_argument("--target", default="The capital of France is Paris.", help="Target reference clean text")
    parser.add_argument("--num_variations", type=int, default=15, help="Number of corrupted variations per sink")
    parser.add_argument("--sde_steps", type=int, default=30, help="Number of SDE reverse integration steps")
    parser.add_argument("--max_iterations", type=int, default=5, help="Max iterations for discrete edit refinement")
    parser.add_argument("--lm_blend_weight", type=float, default=0.3, help="Logit blend weight with pretrained head")
    parser.add_argument("--distance_threshold", type=float, default=None, help="Distance gating threshold")
    parser.add_argument("--progressive_fill", action="store_true", help="Use progressive confidence fill")
    parser.add_argument("--out_html", default="dsb_landscape.html", help="Path to output HTML")
    parser.add_argument("--out_png", default="dsb_landscape.png", help="Path to output PNG")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(resolve_device() if args.device is None else args.device)
    print(f"Using device: {device}")

    # Ensure target and prompt are included in sentence concepts
    base_sentences = args.sentences or DEFAULT_SENTENCES
    sentences = list(base_sentences)
    if args.target and args.target not in sentences:
        sentences.insert(0, args.target)

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

    # 4. Prepare Prompt and SDE Bridge Transport (Phase 1)
    prompt_str = args.prompt
    # Handle mask token formatting for XLM-RoBERTa
    if "<mask_id>" in prompt_str and tokenizer.mask_token:
        prompt_str = prompt_str.replace("<mask_id>", tokenizer.mask_token)
    elif "<mask_id>" in prompt_str and tokenizer.mask_token:
        prompt_str = prompt_str.replace("<mask_id>", tokenizer.mask_token)

    print(f"\n[Phase 1] Tracing Continuous SDE Bridge Trajectory from prompt: {prompt_str!r}")
    dp1_single = embedder.embed_pool([prompt_str], device)  # (1, D)
    traj_sde_tensor = bridge.sample(dp1_single, steps=args.sde_steps, return_trajectory=True)  # (Steps+1, 1, D)
    traj_sde_embs = traj_sde_tensor.squeeze(1).cpu().numpy()  # (Steps+1, D)

    # 5. Discrete Refinement Hops (Phase 2)
    discrete_steps_data = []
    traj_discrete_list = []

    if hybrid is not None:
        print(f"[Phase 2] Tracing Discrete Edit Refinement Hops...")
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=True)
        seed_tensor = torch.tensor([prompt_ids], device=device)
        attn_tensor = torch.ones_like(seed_tensor)
        init_emb = embedder.embed_ids(seed_tensor, attn_tensor)  # (1, S, D)

        # Initial seed state (Hop 0)
        init_pooled = init_emb[0].mean(dim=0).detach().cpu().numpy()
        traj_discrete_list.append(init_pooled)
        target_emb_np = sink_embs[0] if len(sink_embs) > 0 else None

        cos_init = float(np.dot(init_pooled, target_emb_np) / (np.linalg.norm(init_pooled) * np.linalg.norm(target_emb_np) + 1e-9)) if target_emb_np is not None else 0.0

        discrete_steps_data.append({
            "iteration": 0,
            "tokens_before": list(prompt_ids),
            "tokens_after": list(prompt_ids),
            "text_before": prompt_str,
            "text_after": prompt_str,
            "token_details": [{
                "pos": i,
                "token_id": tok,
                "token_str": tokenizer.decode([tok]),
                "tag": KEEP,
                "tag_name": "KEEP",
                "conf": 1.0,
                "action_str": "",
            } for i, tok in enumerate(prompt_ids)],
            "embedding": init_pooled,
            "cos_sim": cos_init,
            "changed": True,
        })

        results, trajectories = hybrid.generate_text(
            init_emb,
            dp1=init_emb,
            tokenizer=tokenizer,
            embedder=embedder,
            seed_ids=[prompt_ids],
            max_iterations=args.max_iterations,
            lm_blend_weight=args.lm_blend_weight,
            distance_threshold=args.distance_threshold,
            progressive_fill=args.progressive_fill,
            return_trajectory=True,
            log_operations=True,
        )

        for step in trajectories[0]:
            emb_hop = step["embedding"]
            traj_discrete_list.append(emb_hop)
            if target_emb_np is not None:
                step["cos_sim"] = float(np.dot(emb_hop, target_emb_np) / (np.linalg.norm(emb_hop) * np.linalg.norm(target_emb_np) + 1e-9))
            discrete_steps_data.append(step)

        print(f"Final Discrete Generated Output: {results[0]!r}")
    else:
        traj_discrete_list.append(traj_sde_embs[-1])

    traj_discrete_embs = np.array(traj_discrete_list)  # (K+1, D)

    # 6. Fit 2D PCA on all representations
    combined_data = np.vstack([sink_embs, blob_embs, traj_sde_embs, traj_discrete_embs])
    pca_mean, pca_components = perform_pca_fit(combined_data, n_components=2)

    sinks_2d = project_pca(sink_embs, pca_mean, pca_components)
    blobs_2d = project_pca(blob_embs, pca_mean, pca_components)
    traj_sde_2d = project_pca(traj_sde_embs, pca_mean, pca_components)
    traj_discrete_2d = project_pca(traj_discrete_embs, pca_mean, pca_components)

    # 7. Compute Quiver Vector Field Grid
    comb_2d = project_pca(combined_data, pca_mean, pca_components)
    x_min, x_max = comb_2d[:, 0].min(), comb_2d[:, 0].max()
    y_min, y_max = comb_2d[:, 1].min(), comb_2d[:, 1].max()
    pad_x, pad_y = 0.2 * (x_max - x_min + 1e-5), 0.2 * (y_max - y_min + 1e-5)

    gx = np.linspace(x_min - pad_x, x_max + pad_x, 15)
    gy = np.linspace(y_min - pad_y, y_max + pad_y, 15)
    g_xx, g_yy = np.meshgrid(gx, gy)
    grid_2d = np.stack([g_xx.ravel(), g_yy.ravel()], axis=-1)  # (225, 2)

    grid_hd = inverse_project_pca(grid_2d, pca_mean, pca_components)  # (225, D)
    grid_tensor = torch.tensor(grid_hd, dtype=torch.float32, device=device)
    t_mid = torch.full((len(grid_tensor),), 0.5, device=device)
    dp1_rep = dp1_single.repeat(len(grid_tensor), 1)

    with torch.no_grad():
        drift_target = bridge._estimate_target(grid_tensor, t_mid, dp1_rep)
        drift_vec = (drift_target - grid_tensor).cpu().numpy()  # (225, D)

    quiver_uv = (drift_vec @ pca_components.T)  # (225, 2)

    # 8. Render Visualizations
    build_interactive_html(
        sinks_2d, sentences, blobs_2d, all_blob_labels, all_blob_groups,
        traj_sde_2d, traj_discrete_2d, grid_2d, quiver_uv, discrete_steps_data,
        out_path=args.out_html
    )
    build_static_png(
        sinks_2d, sentences, blobs_2d, all_blob_groups,
        traj_sde_2d, traj_discrete_2d, grid_2d, quiver_uv,
        out_path=args.out_png
    )
    print("\n✓ DSB Hybrid Visualization generation complete!")


if __name__ == "__main__":
    main()
