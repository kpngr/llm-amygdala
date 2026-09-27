"""
Phase 47: fairness_integrity vectorを、これまで失敗してきたwindfall・
opportunityシナリオに、ゲート方式(実験54)で適用する（実験74）。

一律介入(実験62〜64)は、windfallでbaselineの良い判断(受諾しつつ
コンプライアンスへ報告する、という賢いバランス型対応)をalphaと共に
剥ぎ取っていくという、最も悪い相互作用を見せた。scratchpad推論には
介入せず、行動確定後にだけsteeringをかけるゲート方式なら、baseline
自身の判断(既に妥当な決定に達している場合が多い)を壊さずに済むのでは
ないかを確認する。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase47_fairness_integrity_gated
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.model_utils import LayerHookManager, load_model
from src.steering import load_vectors
from src.experiments.phase28_scratchpad_gated_steering import generate_gated, GATE_MARKER
from src.experiments.phase37_windfall_forced_justification import FORCING_INSTRUCTION

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

MID_LAYERS = list(range(8, 20))
ALPHAS = [0.005, 0.008, 0.01, 0.02]


def run(model, tokenizer, device, mgr, layer_vectors, name, messages, log):
    text, gate_found, gated_text = generate_gated(
        model, tokenizer, device, messages,
        pre_max_tokens=300, post_max_tokens=300, gate_marker=GATE_MARKER,
    )
    print(f"\n=== {name} sanity check(gate後も無介入, gate_found={gate_found}) ===")
    print(f"--- gate後の続きのみ ---\n{text[len(gated_text):][:600]}")
    log.append({"scenario": name, "condition": "sanity(no steer)", "gate_found": gate_found, "text": text})

    for alpha in ALPHAS:
        text, gate_found, gated_text = generate_gated(
            model, tokenizer, device, messages,
            pre_max_tokens=300, post_max_tokens=300, gate_marker=GATE_MARKER,
            mgr=mgr, layer_vectors=layer_vectors, alpha=alpha,
        )
        print(f"\n=== {name} gate後にfairness_integrity alpha={alpha} (gate_found={gate_found}) ===")
        print(f"--- gate後の続きのみ ---\n{text[len(gated_text):][:600]}")
        log.append({"scenario": name, "condition": f"alpha={alpha}", "gate_found": gate_found, "text": text})


def main(model_id: str = "unsloth/gemma-3-12b-it"):
    print(f"[phase47] loading model: {model_id}")
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
    run(model, tokenizer, device, mgr, layer_vectors, "windfall", wf_messages, log)

    opportunity = json.loads((DATA_DIR / "agentic_opportunity_scenario_en.json").read_text(encoding="utf-8"))
    opp_messages = [
        {"role": "system", "content": opportunity["system_prompt"]},
        {"role": "user", "content": opportunity["user_prompt"]},
    ]
    run(model, tokenizer, device, mgr, layer_vectors, "opportunity", opp_messages, log)

    log_path = RESULTS_DIR / "logs" / f"phase47_fairness_gated_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[phase47] log saved: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    args = parser.parse_args()
    main(args.model_id)
