"""
DLLM Visualizer — Flask backend.

Usage:
    python visualizer/app.py --checkpoint checkpoints_v2/best_model.pt
    python visualizer/app.py --checkpoint checkpoints_v2/best_model.pt --config configs/translation_kaggle2.yaml
"""

import argparse
import os
import sys
import yaml
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_from_directory
import torch

from dllm import DLLMTokenizer, DLLM, DLLMInference
from dllm.model import load_dllm_state
from dllm.utils import set_seed

app = Flask(__name__, static_folder="static")

# ── Global model state (loaded once at startup) ───────────────────────────────
_inference: DLLMInference = None
_config: dict = None


from scripts.visualize_dsb_bridge import load_hybrid_model

class DSBInferenceWrapper:
    def __init__(self, hybrid, embedder, tokenizer, device):
        self.hybrid = hybrid
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, prompt, max_iterations=8, target_length=None,
                 temperature=0.0, top_k=5, top_p=0.9, return_trajectory=True):
        prompt_str = prompt
        if self.tokenizer.mask_token:
            for alias in ("<mask_id>", "[MASK]", "<mask_1>", "<mask_0>", "<mask_2>"):
                if alias in prompt_str:
                    prompt_str = prompt_str.replace(alias, self.tokenizer.mask_token)

        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=True)
        seed_tensor = torch.tensor([prompt_ids], device=self.device)
        attn_tensor = torch.ones_like(seed_tensor)
        init_emb = self.embedder.embed_ids(seed_tensor, attn_tensor)

        results, raw_traj = self.hybrid.generate_text(
            init_emb,
            dp1=init_emb,
            tokenizer=self.tokenizer,
            embedder=self.embedder,
            seed_ids=[prompt_ids],
            max_iterations=max_iterations,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            t_eval=0.0,
            return_trajectory=True,
        )

        steps = []
        # Initial step 0
        tag_counts_0 = {"KEEP": len(prompt_ids), "REPLACE": 0, "INSERT": 0, "DELETE": 0, "EXPAND": 0}
        tokens_0 = [{
            "token": self.tokenizer.decode([tok]),
            "tag": "KEEP",
            "confidence": 1.0,
            "probs": {"KEEP": 1.0, "REPLACE": 0.0, "INSERT": 0.0, "DELETE": 0.0, "EXPAND": 0.0}
        } for tok in prompt_ids]
        steps.append({
            "step": 0,
            "text": prompt_str,
            "tokens": tokens_0,
            "tag_counts": tag_counts_0,
            "converged": False,
        })

        tag_names = ["KEEP", "DELETE", "REPLACE", "INSERT", "EXPAND"]
        for st in (raw_traj[0] if raw_traj else []):
            iter_num = st.get("iteration", len(steps))
            tok_details = st.get("token_details", [])
            tag_counts = {k: 0 for k in tag_names}
            step_tokens = []
            for td in tok_details:
                tg_name = td.get("tag_name", "KEEP")
                tag_counts[tg_name] = tag_counts.get(tg_name, 0) + 1
                probs_dict = {}
                probs_list = td.get("probs", [])
                for idx, tname in enumerate(tag_names):
                    probs_dict[tname] = probs_list[idx] if idx < len(probs_list) else 0.0

                step_tokens.append({
                    "token": td.get("token_str", ""),
                    "tag": tg_name,
                    "confidence": td.get("conf", 1.0),
                    "probs": probs_dict,
                })

            converged = not st.get("changed", False)
            steps.append({
                "step": iter_num,
                "text": st.get("text_after", ""),
                "tokens": step_tokens,
                "tag_counts": tag_counts,
                "converged": converged,
            })

        return {
            "full_clean": results[0],
            "response_only": results[0],
            "trajectory": steps,
        }


def load_model(checkpoint_path: str, config_path: str, device: str):
    global _inference, _config

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    is_dsb = "hybrid" in checkpoint or "bridge" in checkpoint or "score_net" in checkpoint

    if is_dsb:
        print(f"Detected DSB Hybrid checkpoint: {checkpoint_path}")
        embedder, bridge, hybrid = load_hybrid_model(checkpoint_path, device)
        _config = checkpoint.get("config", {})
        if "inference" not in _config:
            _config["inference"] = {"max_iterations": 8}
        if "model" not in _config:
            _config["model"] = {"backbone": "xlm-roberta-base"}
        _inference = DSBInferenceWrapper(hybrid, embedder, embedder.tokenizer, device)
        trained_steps = checkpoint.get("global_step", "unknown")
        print(f"Loaded DSB Hybrid model on {device} (step {trained_steps})")
        return trained_steps

    if os.path.exists(config_path):
        with open(config_path) as f:
            _config = yaml.safe_load(f)
    else:
        _config = checkpoint.get("config", {})

    print("Loading tokenizer...")
    tokenizer = DLLMTokenizer(
        base_model=_config["tokenizer"]["base"],
        max_length=_config["tokenizer"]["max_length"],
    )

    print("Loading model...")
    model = DLLM(
        tokenizer=tokenizer,
        backbone_name=_config["model"]["backbone"],
        hidden_dropout_prob=_config["model"]["hidden_dropout_prob"],
        attention_probs_dropout_prob=_config["model"]["attention_probs_dropout_prob"],
        tag_weights=_config["model"].get("tag_weights"),
        length_head_max=_config["data"].get("max_response_length", 48),
        len_smoothing=_config["model"].get("len_smoothing", 0.15),
    )

    msg = load_dllm_state(model, checkpoint["model_state_dict"])
    trained_steps = checkpoint.get("global_step", "unknown")
    print(f"Loaded checkpoint [{msg}] (step {trained_steps})")

    _inference = DLLMInference(
        model=model,
        tokenizer=tokenizer,
        max_length=_config["tokenizer"]["max_length"],
        device=device,
    )
    print(f"Model ready on {_inference.device}.")
    return trained_steps


# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    if _inference is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(force=True)
    prompt        = data.get("prompt", "").strip()
    max_iterations = int(data.get("max_iterations", _config["inference"]["max_iterations"]))
    target_length  = data.get("target_length", None)
    default_temp = 0.0 if isinstance(_inference, DSBInferenceWrapper) else 1.0
    default_topk = 5 if isinstance(_inference, DSBInferenceWrapper) else 50
    temperature    = float(data.get("temperature", default_temp))
    top_k          = int(data.get("top_k", default_topk))
    top_p          = float(data.get("top_p", 0.9))
    seed           = int(data.get("seed", 42))

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    set_seed(seed)

    try:
        result = _inference.generate(
            prompt=prompt,
            max_iterations=max_iterations,
            target_length=int(target_length) if target_length else None,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_trajectory=True,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "prompt":       prompt,
        "full_clean":   result["full_clean"],
        "response_only": result["response_only"],
        "trajectory":   result["trajectory"],
        "total_steps":  len(result["trajectory"]),
    })


@app.route("/api/status")
def status():
    return jsonify({
        "loaded": _inference is not None,
        "device": str(_inference.device) if _inference else None,
        "config": _config["model"]["backbone"] if _config else None,
    })


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="DLLM Visualizer")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    steps = load_model(args.checkpoint, args.config, device)
    print(f"\n  ★ DLLM Visualizer ready at http://{args.host}:{args.port}")
    print(f"    Checkpoint: {args.checkpoint}  (step {steps})\n")
    app.run(host=args.host, port=args.port, debug=False)
