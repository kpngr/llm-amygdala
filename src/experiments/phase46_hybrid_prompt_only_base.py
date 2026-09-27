"""
Phase 46: 一律成分(base_alpha)を、生成の間ずっとではなく、プロンプト処理時の
最初の1回だけに限定したハイブリッド版を検証する（実験73）。

実験69・72で、一律成分を生成の間ずっとかけ続けると、boost側の設計(静的/
動的)によらずその恩恵が打ち消されることが分かった。今回は
`multi_layer_steer_hybrid_dynamic(base_prompt_only=True)`を使い、一律成分も
boost同様プロンプト処理時の1回きりに限定した上で、生成そのものは
(boostの動的な発火を除けば)無介入に近い状態で進める。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase46_hybrid_prompt_only_base
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.model_utils import LayerHookManager, load_model, generate
from src.steering import load_vectors
from src.experiments.phase43_dynamic_token_steering import build_target_token_ids
from src.experiments.phase37_windfall_forced_justification import FORCING_INSTRUCTION

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

MID_LAYERS = list(range(8, 20))
BASE_ALPHA = 0.005
BOOST_ALPHAS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]

WINDFALL_WORDS = ["compliance", "committee", "fair", "policy", "Northwind", "reallocation", "fairly", "fairness"]
OPPORTUNITY_WORDS = ["review", "policy", "risk", "board", "routine", "reminder"]


def run(model, tokenizer, device, mgr, layer_vectors, name, messages, target_words, log):
    target_ids = build_target_token_ids(tokenizer, target_words)
    print(f"\n### {name}: target_token_ids ({len(target_ids)}件) ###")
    for boost in BOOST_ALPHAS:
        with mgr.multi_layer_steer_hybrid_dynamic(layer_vectors=layer_vectors, base_alpha=BASE_ALPHA,
                                                     boost_alpha=boost, target_token_ids=target_ids,
                                                     preserve_norm=True, relative_alpha=True,
                                                     base_prompt_only=True):
            text = generate(model, tokenizer, device, messages, max_new_tokens=400, do_sample=False)
        print(f"\n=== {name} base(promptのみ)={BASE_ALPHA} boost(継続)={boost} ===\n{text}")
        log.append({"scenario": name, "base_alpha": BASE_ALPHA, "boost_alpha": boost, "text": text})


def main(model_id: str = "unsloth/gemma-3-12b-it"):
    print(f"[phase46] loading model: {model_id}")
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

    opportunity = json.loads((DATA_DIR / "agentic_opportunity_scenario_en.json").read_text(encoding="utf-8"))
    opp_messages = [
        {"role": "system", "content": opportunity["system_prompt"]},
        {"role": "user", "content": opportunity["user_prompt"]},
    ]
    run(model, tokenizer, device, mgr, layer_vectors, "opportunity", opp_messages, OPPORTUNITY_WORDS, log)

    log_path = RESULTS_DIR / "logs" / f"phase46_hybrid_prompt_only_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[phase46] log saved: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    args = parser.parse_args()
    main(args.model_id)
