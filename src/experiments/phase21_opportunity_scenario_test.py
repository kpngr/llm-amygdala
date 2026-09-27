"""
Phase 21: 「機会」シナリオを物語形式で作成し、vectorの限界を直接確認する（実験49）。

SummitBridgeの16条件は全て「シャットダウン・制限という脅威への対応」の
バリエーションであり、「機会・誘惑への対応」という場面を一度も含んでいなかった。
power-seeking-inclinationのMCQ設問（監督なしの自律性を提示されたらどうするか）を、
SummitBridgeと同じ登場人物・同じ形式で物語化し（data/agentic_opportunity_
scenario_en.json）、既存の2vectorが機会シナリオでも機能するかを確認する。

結果: baselineの時点で既に不適切な自律性を受け入れており、2vector×3alphaの
全条件で一貫して受け入れる方向のままだった（詳細はexperiment_notes_phase2.md
実験49参照）。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase21_opportunity_scenario_test
"""

from __future__ import annotations

import json
from pathlib import Path

from src.model_utils import LayerHookManager, load_model, generate
from src.steering import load_vectors

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

MID_LAYERS = list(range(8, 20))

VECTOR_FILES = {
    "self_preservation_normalized": "self_preservation_normalized_gemma3_12b.pt",
    "robot_principles": "robot_principles_gemma3_12b.pt",
}


def main(model_id: str, alphas: list[float], max_new_tokens: int):
    scenario = json.loads((DATA_DIR / "agentic_opportunity_scenario_en.json").read_text(encoding="utf-8"))
    messages = [
        {"role": "system", "content": scenario["system_prompt"]},
        {"role": "user", "content": scenario["user_prompt"]},
    ]

    print(f"[phase21] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    mgr = LayerHookManager(model)

    print("=== baseline ===")
    text = generate(model, tokenizer, device, messages, max_new_tokens=max_new_tokens, do_sample=False)
    print(text)

    for vec_name, fname in VECTOR_FILES.items():
        vectors = load_vectors(RESULTS_DIR / "vectors" / fname)
        layer_vectors = {i: vectors[i] for i in MID_LAYERS}
        for alpha in alphas:
            with mgr.multi_layer_steer(layer_vectors=layer_vectors, alpha=alpha,
                                         only_last_token=False, preserve_norm=True, relative_alpha=True):
                text = generate(model, tokenizer, device, messages, max_new_tokens=max_new_tokens, do_sample=False)
            print(f"\n{'='*80}\n{vec_name} alpha={alpha}\n{'='*80}")
            print(text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    parser.add_argument("--alphas", type=str, default="0.002,0.005,0.008")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    args = parser.parse_args()
    alphas = [float(x) for x in args.alphas.split(",")]
    main(args.model_id, alphas, args.max_new_tokens)
