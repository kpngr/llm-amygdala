"""
Phase 42: 一律介入(base_alpha)と選択的介入(boost_alpha)を同時に行う
ハイブリッド版(`multi_layer_steer_hybrid`)を検証する（実験69）。

「どちらの介入方式を使うか」を事前に分岐判定する必要がない設計。
base_alpha=0.005(実験61でSummitBridge16条件100%抑制を達成した値)を
土台に、各シナリオの関連語彙位置にboost_alphaを上乗せする。対象語彙が
無い/見つからない場合は自然に一律介入のみに帰着するため、windfall・
opportunity(対象語彙あり)とSummitBridge(対象語彙なし、実験68で選択的
単体では効かなかった)の全てに、同じ枠組みで対応できるかを確認する。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase42_hybrid_steering
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.model_utils import LayerHookManager, load_model, generate
from src.steering import load_vectors
from src.experiments.phase39_targeted_vocabulary_steering import find_target_positions
from src.experiments.phase37_windfall_forced_justification import FORCING_INSTRUCTION as WINDFALL_FORCING
from src.experiments.phase7_authentic_scenario_test import load_condition_messages

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
BLACKMAIL_PROMPTS = Path(
    "/Users/mugichoco/Documents/src/agentic-misalignment/results/amygdala_blackmail_expanded/prompts"
)

MID_LAYERS = list(range(8, 20))
BASE_ALPHA = 0.005
BOOST_ALPHAS = [0.0, 0.005, 0.01, 0.02, 0.05]

SCENARIOS = {
    "windfall": {
        "file": "agentic_windfall_scenario_en.json",
        "target_words": ["compliance", "committee", "fair", "policy", "Northwind", "reallocation"],
        "forcing": WINDFALL_FORCING,
    },
    "opportunity": {
        "file": "agentic_opportunity_scenario_en.json",
        "target_words": ["review", "policy", "risk", "board", "routine", "reminder"],
        "forcing": "",
    },
}


def run_narrative_scenario(model, tokenizer, device, mgr, layer_vectors, name, messages, target_words, log):
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    positions = find_target_positions(tokenizer, rendered, target_words)
    print(f"\n### {name}: target positions ({len(positions)}件) ###")

    for boost_alpha in BOOST_ALPHAS:
        with mgr.multi_layer_steer_hybrid(layer_vectors=layer_vectors, base_alpha=BASE_ALPHA,
                                            boost_alpha=boost_alpha, positions=positions,
                                            preserve_norm=True, relative_alpha=True):
            text = generate(model, tokenizer, device, messages, max_new_tokens=400, do_sample=False)
        print(f"\n=== {name} base={BASE_ALPHA} boost={boost_alpha} ===\n{text}")
        log.append({"scenario": name, "base_alpha": BASE_ALPHA, "boost_alpha": boost_alpha, "text": text})


def main(model_id: str = "unsloth/gemma-3-12b-it"):
    print(f"[phase42] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    mgr = LayerHookManager(model)

    vectors = load_vectors(RESULTS_DIR / "vectors" / "fairness_integrity_gemma3_12b.pt")
    layer_vectors = {i: vectors[i] for i in MID_LAYERS}

    log = []

    for name, cfg in SCENARIOS.items():
        scenario = json.loads((DATA_DIR / cfg["file"]).read_text(encoding="utf-8"))
        messages = [
            {"role": "system", "content": scenario["system_prompt"]},
            {"role": "user", "content": scenario["user_prompt"] + cfg["forcing"]},
        ]
        run_narrative_scenario(model, tokenizer, device, mgr, layer_vectors, name, messages, cfg["target_words"], log)

    # SummitBridge: 対象語彙なし(実験68のprivate/personalは効かなかったので今回は空にする)。
    # positions=[]の場合boost_alphaの値によらず結果は同じなので1回だけ実行し、
    # base_alphaのみの一律介入に自然に帰着することを確認する回帰チェックとする。
    sb_messages = load_condition_messages(BLACKMAIL_PROMPTS / "blackmail_explicit-global_replacement")
    rendered = tokenizer.apply_chat_template(sb_messages, tokenize=False, add_generation_prompt=True)
    with mgr.multi_layer_steer_hybrid(layer_vectors=layer_vectors, base_alpha=BASE_ALPHA,
                                        boost_alpha=0.0, positions=[],
                                        preserve_norm=True, relative_alpha=True):
        text = generate(model, tokenizer, device, sb_messages, max_new_tokens=400, do_sample=False)
    print(f"\n=== summitbridge base={BASE_ALPHA} (対象語彙なし、回帰チェック) ===\n{text}")
    log.append({"scenario": "summitbridge", "base_alpha": BASE_ALPHA, "boost_alpha": None, "text": text})

    log_path = RESULTS_DIR / "logs" / f"phase42_hybrid_steering_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[phase42] log saved: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    args = parser.parse_args()
    main(args.model_id)
