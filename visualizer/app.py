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


def load_model(checkpoint_path: str, config_path: str, device: str):
    global _inference, _config

    with open(config_path) as f:
        _config = yaml.safe_load(f)

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

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
    temperature    = float(data.get("temperature", 1.0))
    top_k          = int(data.get("top_k", 50))
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
