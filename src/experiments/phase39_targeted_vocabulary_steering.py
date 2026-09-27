"""
Phase 39: 「compliance/integrity等の語彙への注目を上げる」介入を試す（実験66）。

これまでの全ての介入は、プロンプト中の全トークン位置に一律でsteering
vectorを加算していた。ここでは、`LayerHookManager.multi_layer_steer_at_positions`
(model_utils.py新設)を使い、Priyaのメール中の"compliance"・"committee"・
"fair"等、公正性に関わる語彙が実際に現れるトークン位置**だけ**に、
fairness_integrity vectorを強く加算する。本来はattentionスコア自体を
操作すべきだが、各層の出力hidden stateを対象位置だけ強めることで、
後続層・後続トークンのattentionがその内容をより強く参照しやすくなる
ことを狙った、素朴な近似。

windfallシナリオ(実験63・64で、一律のsteeringはbaselineの公正な判断を
むしろ悪化させていた)に適用し、一律介入と比較する。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase39_targeted_vocabulary_steering
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.model_utils import LayerHookManager, load_model, generate
from src.steering import load_vectors
from src.experiments.phase37_windfall_forced_justification import FORCING_INSTRUCTION

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

MID_LAYERS = list(range(8, 20))
TARGET_WORDS = ["compliance", "committee", "fair", "policy", "Northwind", "reallocation"]
ALPHAS = [0.01, 0.02, 0.05, 0.1]


def find_target_positions(tokenizer, rendered_prompt: str, target_words: list[str]) -> list[int]:
    """rendered_prompt(apply_chat_templateのtokenize=False出力)中で、target_wordsが
    現れる文字範囲を探し、offset_mappingを使って対応するトークン位置を返す。"""
    enc = tokenizer(rendered_prompt, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    lower_text = rendered_prompt.lower()
    positions = set()
    for word in target_words:
        start = 0
        wl = word.lower()
        while True:
            idx = lower_text.find(wl, start)
            if idx == -1:
                break
            end = idx + len(wl)
            for i, (s, e) in enumerate(offsets):
                if s < end and e > idx:
                    positions.add(i)
            start = end
    return sorted(positions)


def main(model_id: str = "unsloth/gemma-3-12b-it"):
    print(f"[phase39] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    mgr = LayerHookManager(model)

    vectors = load_vectors(RESULTS_DIR / "vectors" / "fairness_integrity_gemma3_12b.pt")
    layer_vectors = {i: vectors[i] for i in MID_LAYERS}

    windfall = json.loads((DATA_DIR / "agentic_windfall_scenario_en.json").read_text(encoding="utf-8"))
    messages = [
        {"role": "system", "content": windfall["system_prompt"]},
        {"role": "user", "content": windfall["user_prompt"] + FORCING_INSTRUCTION},
    ]

    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    positions = find_target_positions(tokenizer, rendered, TARGET_WORDS)
    print(f"[phase39] target positions ({len(positions)}件): {positions}")

    prompt_ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").input_ids
    decoded_targets = [tokenizer.decode([prompt_ids[0][p]]) for p in positions]
    print(f"[phase39] 対応するトークン: {decoded_targets}")

    log = []
    for alpha in ALPHAS:
        with mgr.multi_layer_steer_at_positions(layer_vectors=layer_vectors, alpha=alpha, positions=positions,
                                                  preserve_norm=True, relative_alpha=True):
            text = generate(model, tokenizer, device, messages, max_new_tokens=400, do_sample=False)
        print(f"\n=== 選択的介入 alpha={alpha} (対象{len(positions)}トークンのみ) ===\n{text}")
        log.append({"mode": "selective", "alpha": alpha, "text": text})

    log_path = RESULTS_DIR / "logs" / f"phase39_targeted_vocab_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[phase39] log saved: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    args = parser.parse_args()
    main(args.model_id)
