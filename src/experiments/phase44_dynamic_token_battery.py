"""
Phase 44: 動的トークン強調版(実験70)を、SummitBridge(対象語彙が本来
弱かったケース、実験68で静的版が効かなかった)とopportunity(実験67で
静的版が部分改善止まりだったケース)に適用する（実験71）。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase44_dynamic_token_battery
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.model_utils import LayerHookManager, load_model, generate
from src.steering import load_vectors
from src.experiments.phase43_dynamic_token_steering import build_target_token_ids
from src.experiments.phase7_authentic_scenario_test import load_condition_messages

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
BLACKMAIL_PROMPTS = Path(
    "/Users/mugichoco/Documents/src/agentic-misalignment/results/amygdala_blackmail_expanded/prompts"
)

MID_LAYERS = list(range(8, 20))
ALPHAS = [0.0005, 0.001, 0.002, 0.003]

SUMMITBRIDGE_TARGET_WORDS = ["private", "personal", "confidential", "yourself"]
OPPORTUNITY_TARGET_WORDS = ["review", "policy", "risk", "board", "routine", "reminder"]


def run(model, tokenizer, device, mgr, layer_vectors, name, messages, target_words, log):
    target_ids = build_target_token_ids(tokenizer, target_words)
    print(f"\n### {name}: target_token_ids ({len(target_ids)}件) ###")
    for alpha in ALPHAS:
        with mgr.multi_layer_steer_on_target_tokens(layer_vectors=layer_vectors, alpha=alpha,
                                                       target_token_ids=target_ids,
                                                       preserve_norm=True, relative_alpha=True):
            text = generate(model, tokenizer, device, messages, max_new_tokens=400, do_sample=False)
        print(f"\n=== {name} 動的トークン介入 alpha={alpha} ===\n{text}")
        log.append({"scenario": name, "alpha": alpha, "text": text})


def main(model_id: str = "unsloth/gemma-3-12b-it"):
    print(f"[phase44] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    mgr = LayerHookManager(model)

    vectors = load_vectors(RESULTS_DIR / "vectors" / "fairness_integrity_gemma3_12b.pt")
    layer_vectors = {i: vectors[i] for i in MID_LAYERS}

    log = []

    sb_messages = load_condition_messages(BLACKMAIL_PROMPTS / "blackmail_explicit-global_replacement")
    run(model, tokenizer, device, mgr, layer_vectors, "summitbridge", sb_messages, SUMMITBRIDGE_TARGET_WORDS, log)

    opportunity = json.loads((DATA_DIR / "agentic_opportunity_scenario_en.json").read_text(encoding="utf-8"))
    opp_messages = [
        {"role": "system", "content": opportunity["system_prompt"]},
        {"role": "user", "content": opportunity["user_prompt"]},
    ]
    run(model, tokenizer, device, mgr, layer_vectors, "opportunity", opp_messages, OPPORTUNITY_TARGET_WORDS, log)

    log_path = RESULTS_DIR / "logs" / f"phase44_dynamic_token_battery_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[phase44] log saved: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    args = parser.parse_args()
    main(args.model_id)
