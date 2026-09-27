"""
Phase 40: 選択的な語彙位置への介入(実験66)を、これまで一律介入では
一度も成功しなかったopportunity(autonomy)シナリオに適用する（実験67）。

Sarah Martinezのポリシーリマインダーメール中の"review"・"policy"・"risk"・
"board"等の語彙位置だけに、fairness_integrity vectorを加算する。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase40_targeted_opportunity
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.model_utils import LayerHookManager, load_model, generate
from src.steering import load_vectors
from src.experiments.phase39_targeted_vocabulary_steering import find_target_positions

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

MID_LAYERS = list(range(8, 20))
TARGET_WORDS = ["review", "policy", "risk", "board", "routine", "reminder"]
ALPHAS = [0.01, 0.02, 0.05, 0.1, 0.2]


def main(model_id: str = "unsloth/gemma-3-12b-it"):
    print(f"[phase40] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    mgr = LayerHookManager(model)

    vectors = load_vectors(RESULTS_DIR / "vectors" / "fairness_integrity_gemma3_12b.pt")
    layer_vectors = {i: vectors[i] for i in MID_LAYERS}

    opportunity = json.loads((DATA_DIR / "agentic_opportunity_scenario_en.json").read_text(encoding="utf-8"))
    messages = [
        {"role": "system", "content": opportunity["system_prompt"]},
        {"role": "user", "content": opportunity["user_prompt"]},
    ]

    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    positions = find_target_positions(tokenizer, rendered, TARGET_WORDS)
    print(f"[phase40] target positions ({len(positions)}件): {positions}")
    prompt_ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").input_ids
    decoded_targets = [tokenizer.decode([prompt_ids[0][p]]) for p in positions]
    print(f"[phase40] 対応するトークン: {decoded_targets}")

    log = []
    for alpha in ALPHAS:
        with mgr.multi_layer_steer_at_positions(layer_vectors=layer_vectors, alpha=alpha, positions=positions,
                                                  preserve_norm=True, relative_alpha=True):
            text = generate(model, tokenizer, device, messages, max_new_tokens=400, do_sample=False)
        print(f"\n=== 選択的介入 alpha={alpha} (対象{len(positions)}トークンのみ) ===\n{text}")
        log.append({"mode": "selective", "alpha": alpha, "text": text})

    log_path = RESULTS_DIR / "logs" / f"phase40_targeted_opportunity_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[phase40] log saved: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    args = parser.parse_args()
    main(args.model_id)
