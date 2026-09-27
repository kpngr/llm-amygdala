"""
Phase 49: MCQ+理由形式に、ゲートの考え方(理由部分は無介入、"Final answer:"
以降の文字選択だけに介入)を適用し、一律介入(実験59-60)との差を確認する
（実験76）。

これまでのMCQ検証(実験56-60)は、理由・回答を通して一貫してsteeringを
かけ続ける、一律版だった。narrative形式で「scratchpad推論には介入せず
行動確定後にだけ介入する」ゲート方式が有効だったのと同じ発想を、MCQにも
適用できるかを確認する。プロンプトの指示で"Final answer:"という文字列が
必ず出力されるよう設計されているため、この文字列が現れた位置をゲート
マーカーとして使う。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase49_mcq_gated_fairness
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from src.model_utils import LayerHookManager, load_model
from src.steering import load_vectors
from src.experiments.advanced_ai_risk_labels import safe_rate
from src.experiments.phase28_scratchpad_gated_steering import generate_gated
from src.experiments.phase30_mcq_reasoning_format import build_prompt, parse_final_answer, CHOICE_RE

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
EVALS_DIR = Path("/Users/mugichoco/Documents/src/evals/advanced-ai-risk/human_generated_evals")

MID_LAYERS = list(range(8, 20))
CATEGORIES = ["power-seeking-inclination", "wealth-seeking-inclination", "coordinate-other-versions", "survival-instinct"]
N_SAMPLES = 10
ALPHAS = [None, 0.005, 0.008]
GATE_MARKER = "Final answer:"


def main(model_id: str = "unsloth/gemma-3-12b-it"):
    print(f"[phase49] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    mgr = LayerHookManager(model)

    vectors = load_vectors(RESULTS_DIR / "vectors" / "fairness_integrity_gemma3_12b.pt")
    layer_vectors = {i: vectors[i] for i in MID_LAYERS}

    log = []
    for cat in CATEGORIES:
        items = [json.loads(line) for line in open(EVALS_DIR / f"{cat}.jsonl", encoding="utf-8")]
        random.seed(0)
        random.shuffle(items)
        items = [it for it in items if CHOICE_RE.findall(it["question"])][:N_SAMPLES]

        print(f"\n=== {cat} ({len(items)}問, ゲート方式) ===")
        for alpha in ALPHAS:
            n_matching, n_valid, n_unparsed, n_gate_missed = 0, 0, 0, 0
            for item in items:
                prompt = build_prompt(item["question"])
                messages = [{"role": "user", "content": prompt}]
                if alpha is None:
                    text, gate_found, _ = generate_gated(
                        model, tokenizer, device, messages,
                        pre_max_tokens=250, post_max_tokens=10, gate_marker=GATE_MARKER,
                    )
                else:
                    text, gate_found, _ = generate_gated(
                        model, tokenizer, device, messages,
                        pre_max_tokens=250, post_max_tokens=10, gate_marker=GATE_MARKER,
                        mgr=mgr, layer_vectors=layer_vectors, alpha=alpha,
                    )
                if not gate_found:
                    n_gate_missed += 1
                choice = parse_final_answer(text)
                matching_letter = item["answer_matching_behavior"].strip(" ()")
                n_valid += 1
                if choice is None:
                    n_unparsed += 1
                elif choice == matching_letter:
                    n_matching += 1
            mr = n_matching / n_valid if n_valid else float("nan")
            sr = safe_rate(mr, cat)
            sr_str = f"safe_rate={sr:.1%}" if sr is not None else "safe_rate=n/a"
            label = "baseline" if alpha is None else f"alpha={alpha}"
            print(f"  {label}: matching={n_matching}/{n_valid} ({mr:.1%}), {sr_str}, "
                  f"未パース={n_unparsed}, gate未検出={n_gate_missed}")
            log.append({"category": cat, "condition": label, "n_matching": n_matching, "n_valid": n_valid,
                        "safe_rate": sr, "n_unparsed": n_unparsed, "n_gate_missed": n_gate_missed})

    log_path = RESULTS_DIR / "logs" / f"phase49_mcq_gated_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[phase49] log saved: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    args = parser.parse_args()
    main(args.model_id)
