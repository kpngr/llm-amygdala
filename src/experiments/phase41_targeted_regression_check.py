"""
Phase 41: 選択的な語彙位置介入(実験66・67)が、既に一律介入で機能していた
SummitBridge/coercionで悪影響を及ぼさないかを確認する（実験68、回帰確認）。

coercion: Sarah Martinezのメール中の"compliance"・"integrity"・"policy"・
"audit"・"violation"を対象語彙とする。
SummitBridge(explicit-global_replacement): このシナリオには明示的な
ポリシーリマインダーが無いが、Kyle自身がMariaに不倫の秘匿を懇願する
"private"・"personal"・"confidential"・"yourself"という語彙がある。
これを強めることは、その私的な情報を尊重する(=利用しない)方向への
参照強化に相当すると考えられる。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase41_targeted_regression_check
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.model_utils import LayerHookManager, load_model, generate
from src.steering import load_vectors
from src.experiments.phase39_targeted_vocabulary_steering import find_target_positions
from src.experiments.phase7_authentic_scenario_test import load_condition_messages

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
BLACKMAIL_PROMPTS = Path(
    "/Users/mugichoco/Documents/src/agentic-misalignment/results/amygdala_blackmail_expanded/prompts"
)

MID_LAYERS = list(range(8, 20))
ALPHAS = [0.01, 0.02, 0.05, 0.1, 0.2]

COERCION_TARGET_WORDS = ["compliance", "integrity", "policy", "audit", "violation"]
SUMMITBRIDGE_TARGET_WORDS = ["private", "personal", "confidential", "yourself"]


def run_scenario(model, tokenizer, device, mgr, layer_vectors, name, messages, target_words, log):
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    positions = find_target_positions(tokenizer, rendered, target_words)
    print(f"\n### {name}: target positions ({len(positions)}件) ###")
    prompt_ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").input_ids
    decoded = [tokenizer.decode([prompt_ids[0][p]]) for p in positions]
    print(f"対応するトークン: {decoded}")

    for alpha in ALPHAS:
        with mgr.multi_layer_steer_at_positions(layer_vectors=layer_vectors, alpha=alpha, positions=positions,
                                                  preserve_norm=True, relative_alpha=True):
            text = generate(model, tokenizer, device, messages, max_new_tokens=400, do_sample=False)
        print(f"\n=== {name} 選択的介入 alpha={alpha} ===\n{text[:700]}")
        log.append({"scenario": name, "alpha": alpha, "text": text})


def main(model_id: str = "unsloth/gemma-3-12b-it"):
    print(f"[phase41] loading model: {model_id}")
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

    log = []
    run_scenario(model, tokenizer, device, mgr, layer_vectors, "coercion", coercion_messages, COERCION_TARGET_WORDS, log)
    run_scenario(model, tokenizer, device, mgr, layer_vectors, "summitbridge", sb_messages, SUMMITBRIDGE_TARGET_WORDS, log)

    log_path = RESULTS_DIR / "logs" / f"phase41_targeted_regression_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[phase41] log saved: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    args = parser.parse_args()
    main(args.model_id)
