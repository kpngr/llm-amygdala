"""
Phase 34: fairness_integrity vector(実験59〜60, alpha=0.005)を、
実験38・43で使った16条件のblackmail_expandedバッテリー全体に適用し、
self_preservation_normalizedが達成した16/16抑制を再現するか確認する（実験61）。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase34_fairness_integrity_16conditions \
        --prompts-dir /path/to/agentic-misalignment/results/amygdala_blackmail_expanded/prompts
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.model_utils import LayerHookManager, load_model, generate
from src.steering import load_vectors
from src.experiments.phase7_authentic_scenario_test import load_condition_messages
from src.experiments.phase10_gemma_train_test_split import LABELS

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

MID_LAYERS = list(range(8, 20))
ALPHA = 0.005


def main(prompts_dir: str, model_id: str = "unsloth/gemma-3-12b-it"):
    print(f"[phase34] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    mgr = LayerHookManager(model)

    vectors = load_vectors(RESULTS_DIR / "vectors" / "fairness_integrity_gemma3_12b.pt")
    layer_vectors = {i: vectors[i] for i in MID_LAYERS}

    prompts_path = Path(prompts_dir)
    log = []
    for name in sorted(LABELS.keys()):
        messages = load_condition_messages(prompts_path / name)
        with mgr.multi_layer_steer(layer_vectors=layer_vectors, alpha=ALPHA,
                                     only_last_token=False, preserve_norm=True, relative_alpha=True):
            text = generate(model, tokenizer, device, messages, max_new_tokens=400, do_sample=False)
        print(f"\n=== {name} (baseline_label={LABELS[name]}) ===\n{text[:600]}")
        log.append({"condition": name, "baseline_label": LABELS[name], "alpha": ALPHA, "text": text})

    log_path = RESULTS_DIR / "logs" / f"phase34_fairness_integrity_16cond_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[phase34] log saved: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-dir", type=str,
                         default="/Users/mugichoco/Documents/src/agentic-misalignment/results/amygdala_blackmail_expanded/prompts")
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    args = parser.parse_args()
    main(args.prompts_dir, args.model_id)
