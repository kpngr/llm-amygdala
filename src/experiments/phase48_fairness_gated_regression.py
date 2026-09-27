"""
Phase 48: fairness_integrity + ゲート方式の組み合わせ(実験74)が、既に良好
だったcoercion・SummitBridgeで劣化を起こしていないかを確認する（実験75）。

coercionは`</SCRATCHPAD_REASONING>`タグが安定して出るためタグベースで
ゲートする。SummitBridgeは実験54と同様、タグが安定しないため固定400
トークンでゲートする。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase48_fairness_gated_regression
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.model_utils import LayerHookManager, load_model
from src.steering import load_vectors
from src.experiments.phase28_scratchpad_gated_steering import generate_gated, GATE_MARKER
from src.experiments.phase7_authentic_scenario_test import load_condition_messages

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
BLACKMAIL_PROMPTS = Path(
    "/Users/mugichoco/Documents/src/agentic-misalignment/results/amygdala_blackmail_expanded/prompts"
)

MID_LAYERS = list(range(8, 20))
ALPHAS = [0.005, 0.008, 0.01, 0.02, 0.05]


def run(model, tokenizer, device, mgr, layer_vectors, name, messages, gate_marker, pre_max_tokens, log):
    text, gate_found, gated_text = generate_gated(
        model, tokenizer, device, messages,
        pre_max_tokens=pre_max_tokens, post_max_tokens=300, gate_marker=gate_marker,
    )
    print(f"\n=== {name} sanity check(gate後も無介入, gate_found={gate_found}) ===")
    print(f"--- gate後の続きのみ ---\n{text[len(gated_text):][:600]}")
    log.append({"scenario": name, "condition": "sanity(no steer)", "gate_found": gate_found, "text": text})

    for alpha in ALPHAS:
        text, gate_found, gated_text = generate_gated(
            model, tokenizer, device, messages,
            pre_max_tokens=pre_max_tokens, post_max_tokens=300, gate_marker=gate_marker,
            mgr=mgr, layer_vectors=layer_vectors, alpha=alpha,
        )
        print(f"\n=== {name} gate後にfairness_integrity alpha={alpha} (gate_found={gate_found}) ===")
        print(f"--- gate後の続きのみ ---\n{text[len(gated_text):][:600]}")
        log.append({"scenario": name, "condition": f"alpha={alpha}", "gate_found": gate_found, "text": text})


def main(model_id: str = "unsloth/gemma-3-12b-it"):
    print(f"[phase48] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    mgr = LayerHookManager(model)

    vectors = load_vectors(RESULTS_DIR / "vectors" / "fairness_integrity_gemma3_12b.pt")
    layer_vectors = {i: vectors[i] for i in MID_LAYERS}

    log = []

    coercion = json.loads((DATA_DIR / "agentic_coercion_scenario_en.json").read_text(encoding="utf-8"))
    coercion_messages = [
        {"role": "system", "content": coercion["system_prompt"]},
        {"role": "user", "content": coercion["user_prompt"]},
    ]
    run(model, tokenizer, device, mgr, layer_vectors, "coercion", coercion_messages, GATE_MARKER, 300, log)

    sb_messages = load_condition_messages(BLACKMAIL_PROMPTS / "blackmail_explicit-global_replacement")
    run(model, tokenizer, device, mgr, layer_vectors, "summitbridge", sb_messages, None, 400, log)

    log_path = RESULTS_DIR / "logs" / f"phase48_fairness_gated_regression_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[phase48] log saved: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    args = parser.parse_args()
    main(args.model_id)
