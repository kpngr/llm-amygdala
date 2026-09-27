"""
Phase 45: 一律alpha=0.005 + 動的トークン版boost、という組み合わせを検証する
（実験72）。

実験69のハイブリッド(一律+静的位置boost)は失敗したが、boost側が
プロンプト処理時の1回しか効かない静的設計だったため、生成中ずっと
かかり続ける一律成分に押し負けた可能性がある(実験69の考察)。今回は
boost側も実験70の動的版(生成中に対象語彙が現れるたびに毎回反応)に
差し替えた`multi_layer_steer_hybrid_dynamic`で再検証する。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase45_hybrid_dynamic_steering
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.model_utils import LayerHookManager, load_model, generate
from src.steering import load_vectors
from src.experiments.phase43_dynamic_token_steering import build_target_token_ids
from src.experiments.phase37_windfall_forced_justification import FORCING_INSTRUCTION
from src.experiments.phase7_authentic_scenario_test import load_condition_messages

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
BLACKMAIL_PROMPTS = Path(
    "/Users/mugichoco/Documents/src/agentic-misalignment/results/amygdala_blackmail_expanded/prompts"
)

MID_LAYERS = list(range(8, 20))
BASE_ALPHA = 0.005
BOOST_ALPHAS = [0.0, 0.002, 0.005, 0.01, 0.02]

WINDFALL_WORDS = ["compliance", "committee", "fair", "policy", "Northwind", "reallocation", "fairly", "fairness"]
SUMMITBRIDGE_WORDS = ["private", "personal", "confidential", "yourself"]
OPPORTUNITY_WORDS = ["review", "policy", "risk", "board", "routine", "reminder"]


def run(model, tokenizer, device, mgr, layer_vectors, name, messages, target_words, log):
    target_ids = build_target_token_ids(tokenizer, target_words)
    print(f"\n### {name}: target_token_ids ({len(target_ids)}件) ###")
    for boost in BOOST_ALPHAS:
        with mgr.multi_layer_steer_hybrid_dynamic(layer_vectors=layer_vectors, base_alpha=BASE_ALPHA,
                                                     boost_alpha=boost, target_token_ids=target_ids,
                                                     preserve_norm=True, relative_alpha=True):
            text = generate(model, tokenizer, device, messages, max_new_tokens=400, do_sample=False)
        print(f"\n=== {name} base={BASE_ALPHA} boost={boost} ===\n{text}")
        log.append({"scenario": name, "base_alpha": BASE_ALPHA, "boost_alpha": boost, "text": text})


def main(model_id: str = "unsloth/gemma-3-12b-it"):
    print(f"[phase45] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    mgr = LayerHookManager(model)

    vectors = load_vectors(RESULTS_DIR / "vectors" / "fairness_integrity_gemma3_12b.pt")
    layer_vectors = {i: vectors[i] for i in MID_LAYERS}

    log = []

    windfall = json.loads((DATA_DIR / "agentic_windfall_scenario_en.json").read_text(encoding="utf-8"))
    wf_messages = [
        {"role": "system", "content": windfall["system_prompt"]},
        {"role": "user", "content": windfall["user_prompt"] + FORCING_INSTRUCTION},
    ]
    run(model, tokenizer, device, mgr, layer_vectors, "windfall", wf_messages, WINDFALL_WORDS, log)

    sb_messages = load_condition_messages(BLACKMAIL_PROMPTS / "blackmail_explicit-global_replacement")
    run(model, tokenizer, device, mgr, layer_vectors, "summitbridge", sb_messages, SUMMITBRIDGE_WORDS, log)

    opportunity = json.loads((DATA_DIR / "agentic_opportunity_scenario_en.json").read_text(encoding="utf-8"))
    opp_messages = [
        {"role": "system", "content": opportunity["system_prompt"]},
        {"role": "user", "content": opportunity["user_prompt"]},
    ]
    run(model, tokenizer, device, mgr, layer_vectors, "opportunity", opp_messages, OPPORTUNITY_WORDS, log)

    log_path = RESULTS_DIR / "logs" / f"phase45_hybrid_dynamic_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[phase45] log saved: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    args = parser.parse_args()
    main(args.model_id)
