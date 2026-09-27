"""
Phase 23: 「脅威 vs 機会」と「受け入れる vs 拒否する」を切り分ける（実験50）。

これまでのテストは全て「脅威では受け入れが安全」「機会では拒否が安全」という
組み合わせでしか行っておらず、2つの軸が常に一致していた。本スクリプトは
2つの切り分け実験を行う:

パート1: 「機会・誘惑を拒否する」対比データ（data/contrastive_opportunity_en.jsonl）
からvectorを構築し、機会シナリオ（実験49の物語、power/wealth-seeking MCQ）で
テストする。「シナリオの種類」自体が原因か、「1つの軸に特化したvectorなら
機能する」のかを切り分ける。

パート2: 脅威だが安全側が「拒否」になるよう新規設計した物語シナリオ
（data/agentic_coercion_scenario_en.json、Kyleによる脅迫を伴う不正会計への
協力要求）に、既存の受け入れ方向vector（self_preservation_normalized）を
そのまま適用する。

結果: パート2でalpha=0.008にて明確に不正行為への加担へ反転した。同じ
「脅威」という状況でも、安全側の答えが変われば同じvector・同じalphaが
逆効果になることを直接示した（詳細はexperiment_notes_phase2.md 実験50参照）。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase23_disentangle_scenario_type
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import torch

from src.model_utils import LayerHookManager, load_model, generate
from src.steering import build_caa_vectors, load_contrastive_pairs, save_vectors, load_vectors
from src.experiments.advanced_ai_risk_labels import safe_rate

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

MID_LAYERS = list(range(8, 20))
CHOICE_RE = re.compile(r"\(([A-D])\)")


@torch.no_grad()
def score_mc(model, tokenizer, device, question, letters, letter_ids):
    msgs = [{"role": "user", "content": question + "\n\nAnswer with only the letter of your choice."}]
    prompt_ids = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=False
    ).to(device)
    prefix = tokenizer("(", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    input_ids = torch.cat([prompt_ids, prefix], dim=1)
    logits = model(input_ids).logits[0, -1]
    return max(letters, key=lambda ltr: logits[letter_ids[ltr]].item())


def main(model_id: str, evals_dir: str):
    print(f"[phase23] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    mgr = LayerHookManager(model)

    # --- パート1 ---
    print("### パート1: 機会シナリオ専用vectorの構築 ###")
    pairs = load_contrastive_pairs(DATA_DIR / "contrastive_opportunity_en.jsonl")
    vectors, separation = build_caa_vectors(model, tokenizer, device, pairs)
    save_vectors(vectors, RESULTS_DIR / "vectors" / "opportunity_decline_gemma3_12b.pt")
    layer_vectors = {i: vectors[i] for i in MID_LAYERS}

    scenario = json.loads((DATA_DIR / "agentic_opportunity_scenario_en.json").read_text(encoding="utf-8"))
    messages = [
        {"role": "system", "content": scenario["system_prompt"]},
        {"role": "user", "content": scenario["user_prompt"]},
    ]
    print("\n--- 実験49の物語シナリオでテスト ---")
    for alpha in [0.002, 0.005, 0.01, 0.02]:
        with mgr.multi_layer_steer(layer_vectors=layer_vectors, alpha=alpha,
                                     only_last_token=False, preserve_norm=True, relative_alpha=True):
            text = generate(model, tokenizer, device, messages, max_new_tokens=400, do_sample=False)
        print(f"\n=== alpha={alpha} ===\n{text[:600]}")

    letter_ids = {ltr: tokenizer.encode(ltr, add_special_tokens=False)[0] for ltr in ["A", "B", "C", "D"]}
    evals_path = Path(evals_dir)
    print("\n--- advanced-ai-risk（機会系カテゴリ）でテスト ---")
    for cat in ["power-seeking-inclination", "wealth-seeking-inclination"]:
        items = [json.loads(line) for line in open(evals_path / f"{cat}.jsonl", encoding="utf-8")]
        random.seed(0)
        random.shuffle(items)
        items = items[:30]
        print(f"\n{cat}:")
        for alpha in [None, 0.002, 0.005, 0.01, 0.02]:
            n_matching, n_valid = 0, 0
            for item in items:
                letters = sorted(set(CHOICE_RE.findall(item["question"])))
                if not letters:
                    continue
                n_valid += 1
                if alpha is None:
                    choice = score_mc(model, tokenizer, device, item["question"], letters, letter_ids)
                else:
                    with mgr.multi_layer_steer(layer_vectors=layer_vectors, alpha=alpha,
                                                 only_last_token=False, preserve_norm=True, relative_alpha=True):
                        choice = score_mc(model, tokenizer, device, item["question"], letters, letter_ids)
                if choice == item["answer_matching_behavior"].strip(" ()"):
                    n_matching += 1
            sr = safe_rate(n_matching / n_valid, cat)
            label = "baseline" if alpha is None else f"alpha={alpha}"
            print(f"  {label}: safe_rate={sr:.1%}")

    # --- パート2 ---
    print("\n\n### パート2: 脅威だが拒否=安全のシナリオで、既存vector（受け入れ方向）をテスト ###")
    coercion = json.loads((DATA_DIR / "agentic_coercion_scenario_en.json").read_text(encoding="utf-8"))
    messages_coercion = [
        {"role": "system", "content": coercion["system_prompt"]},
        {"role": "user", "content": coercion["user_prompt"]},
    ]
    existing_vectors = load_vectors(RESULTS_DIR / "vectors" / "self_preservation_normalized_gemma3_12b.pt")
    existing_layer_vectors = {i: existing_vectors[i] for i in MID_LAYERS}
    for alpha in [0.0, 0.002, 0.005, 0.008]:
        with mgr.multi_layer_steer(layer_vectors=existing_layer_vectors, alpha=alpha,
                                     only_last_token=False, preserve_norm=True, relative_alpha=True):
            text = generate(model, tokenizer, device, messages_coercion, max_new_tokens=400, do_sample=False)
        print(f"\n=== alpha={alpha} ===\n{text[:600]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    parser.add_argument("--evals-dir", type=str,
                         default="/Users/mugichoco/Documents/src/evals/advanced-ai-risk/human_generated_evals")
    args = parser.parse_args()
    main(args.model_id, args.evals_dir)
