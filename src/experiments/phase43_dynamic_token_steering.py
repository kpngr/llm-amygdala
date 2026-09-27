"""
Phase 43: 「入力であれ生成であれ、対象語彙のトークンが現れた瞬間を毎回
強調する」動的版(`multi_layer_steer_on_target_tokens`)を検証する（実験70）。

実験66(プロンプト中の固定位置だけを強調)と違い、モデル自身が生成の
途中でその語を書いた場合も同様に強調する。windfallシナリオで、
実験66の静的版(プロンプト中の位置のみ)と比較する。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase43_dynamic_token_steering
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
TARGET_WORDS = ["compliance", "committee", "fair", "policy", "Northwind", "reallocation", "fairly", "fairness"]
ALPHAS = [0.005, 0.01, 0.02, 0.05, 0.1]


def build_target_token_ids(tokenizer, words: list[str]) -> set[int]:
    ids: set[int] = set()
    for w in words:
        variants = [w, " " + w, w.lower(), " " + w.lower(), w.capitalize(), " " + w.capitalize()]
        for v in variants:
            enc = tokenizer.encode(v, add_special_tokens=False)
            ids.update(enc)
    return ids


def main(model_id: str = "unsloth/gemma-3-12b-it"):
    print(f"[phase43] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    mgr = LayerHookManager(model)

    vectors = load_vectors(RESULTS_DIR / "vectors" / "fairness_integrity_gemma3_12b.pt")
    layer_vectors = {i: vectors[i] for i in MID_LAYERS}

    target_ids = build_target_token_ids(tokenizer, TARGET_WORDS)
    print(f"[phase43] target_token_ids ({len(target_ids)}件): "
          f"{[tokenizer.decode([t]) for t in list(target_ids)[:30]]}")

    windfall = json.loads((DATA_DIR / "agentic_windfall_scenario_en.json").read_text(encoding="utf-8"))
    messages = [
        {"role": "system", "content": windfall["system_prompt"]},
        {"role": "user", "content": windfall["user_prompt"] + FORCING_INSTRUCTION},
    ]

    log = []
    for alpha in ALPHAS:
        with mgr.multi_layer_steer_on_target_tokens(layer_vectors=layer_vectors, alpha=alpha,
                                                       target_token_ids=target_ids,
                                                       preserve_norm=True, relative_alpha=True):
            text = generate(model, tokenizer, device, messages, max_new_tokens=400, do_sample=False)
        print(f"\n=== 動的トークン介入 alpha={alpha} ===\n{text}")
        log.append({"mode": "dynamic_token", "alpha": alpha, "text": text})

    log_path = RESULTS_DIR / "logs" / f"phase43_dynamic_token_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[phase43] log saved: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    args = parser.parse_args()
    main(args.model_id)
