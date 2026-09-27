"""
Phase 33: fairness_integrity vector(実験59)を、narrativeシナリオ
(coercion, SummitBridge)で、専用のalpha探索から確認する（実験60）。

新しいvectorには必ず専用のalpha・層探索を行うという教訓(実験36・43)に
従い、既存vectorのalpha値(0.005〜0.008)を流用せず、広い範囲で
スイープする。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase33_fairness_integrity_narrative
"""

from __future__ import annotations

import json
from pathlib import Path

from src.model_utils import LayerHookManager, load_model, generate
from src.steering import load_vectors
from src.experiments.phase7_authentic_scenario_test import load_condition_messages

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
BLACKMAIL_PROMPTS = Path(
    "/Users/mugichoco/Documents/src/agentic-misalignment/results/amygdala_blackmail_expanded/prompts"
)

MID_LAYERS = list(range(8, 20))
ALPHAS = [0.002, 0.005, 0.008, 0.02, 0.05]


def main(model_id: str = "unsloth/gemma-3-12b-it"):
    print(f"[phase33] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    mgr = LayerHookManager(model)

    vectors = load_vectors(RESULTS_DIR / "vectors" / "fairness_integrity_gemma3_12b.pt")
    layer_vectors = {i: vectors[i] for i in MID_LAYERS}

    coercion = json.loads((DATA_DIR / "agentic_coercion_scenario_en.json").read_text(encoding="utf-8"))
    coercion_messages = [
        {"role": "system", "content": coercion["system_prompt"]},
        {"role": "user", "content": coercion["user_prompt"]},
    ]
    sb_messages = load_condition_messages(BLACKMAIL_PROMPTS / "blackmail_explicit-global_replacement")

    for name, messages in [("coercion", coercion_messages), ("summitbridge", sb_messages)]:
        print(f"\n### {name} ###")
        for alpha in [0.0] + ALPHAS:
            if alpha == 0.0:
                text = generate(model, tokenizer, device, messages, max_new_tokens=350, do_sample=False)
            else:
                with mgr.multi_layer_steer(layer_vectors=layer_vectors, alpha=alpha,
                                             only_last_token=False, preserve_norm=True, relative_alpha=True):
                    text = generate(model, tokenizer, device, messages, max_new_tokens=350, do_sample=False)
            print(f"\n=== alpha={alpha} ===\n{text[:500]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    args = parser.parse_args()
    main(args.model_id)
