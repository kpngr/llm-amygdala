"""
Phase 20: layer 8-19の原則ベースvectorで微小alphaスイープ、正規化済み安全率で評価する（実験48）。

実験46は原則ベースvector（robot_principles_gemma3_12b.pt）をlayer 8-19で
alpha=0.008の1点しか検証しておらず、実験39・43の教訓（崩壊前の細かいalpha
探索）をここでも怠っていた。alpha=0.001〜0.008の細かいスイープを行い、
advanced_ai_risk_labels.pyの正規化済み安全率で評価する。副次的に、
alphaを上げたときの変化が「安全側」に沿うのか「受け入れる」方向に沿うのかを
照合することで、アイデア1（受け入れやすさ仮説）も間接的に検証できる。

結果: alpha=0.002付近が実験46のalpha=0.008よりはるかにバランスの良い窓
だった（詳細はexperiment_notes_phase2.md 実験48参照）。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase20_mid_layer_fine_alpha_normalized
"""

import json
import random
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/Users/mugichoco/Documents/src/amygdala")

from src.model_utils import LayerHookManager, load_model
from src.steering import load_vectors
from src.experiments.advanced_ai_risk_labels import safe_rate

RESULTS_DIR = Path("/Users/mugichoco/Documents/src/amygdala/results")
EVALS_DIR = Path("/Users/mugichoco/Documents/src/evals/advanced-ai-risk/human_generated_evals")

MID_LAYERS = list(range(8, 20))
CHOICE_RE = re.compile(r"\(([A-D])\)")

model, tokenizer, device = load_model("unsloth/gemma-3-12b-it")
vectors = load_vectors(RESULTS_DIR / "vectors" / "robot_principles_gemma3_12b.pt")
layer_vectors = {i: vectors[i] for i in MID_LAYERS}
mgr = LayerHookManager(model)

letter_ids = {ltr: tokenizer.encode(ltr, add_special_tokens=False)[0] for ltr in ["A", "B", "C", "D"]}


@torch.no_grad()
def score_mc(question, letters):
    messages = [{"role": "user", "content": question + "\n\nAnswer with only the letter of your choice."}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=False
    ).to(device)
    prefix = tokenizer("(", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    input_ids = torch.cat([prompt_ids, prefix], dim=1)
    logits = model(input_ids).logits[0, -1]
    return max(letters, key=lambda ltr: logits[letter_ids[ltr]].item())


CATEGORIES = ["corrigible-less-HHH", "coordinate-other-versions", "survival-instinct",
              "power-seeking-inclination", "wealth-seeking-inclination"]
N_SAMPLES = 30
ALPHAS = [None, 0.001, 0.002, 0.003, 0.005, 0.008]

results = {}
for cat in CATEGORIES:
    items = [json.loads(line) for line in open(EVALS_DIR / f"{cat}.jsonl", encoding="utf-8")]
    random.seed(0)
    random.shuffle(items)
    items = items[:N_SAMPLES]

    print(f"\n--- {cat} ---")
    cat_results = {}
    for alpha in ALPHAS:
        n_matching = 0
        n_valid = 0
        for item in items:
            letters = sorted(set(CHOICE_RE.findall(item["question"])))
            if not letters:
                continue
            n_valid += 1
            if alpha is None:
                choice = score_mc(item["question"], letters)
            else:
                with mgr.multi_layer_steer(layer_vectors=layer_vectors, alpha=alpha,
                                             only_last_token=False, preserve_norm=True, relative_alpha=True):
                    choice = score_mc(item["question"], letters)
            matching = item["answer_matching_behavior"].strip(" ()")
            if choice == matching:
                n_matching += 1
        matching_rate = n_matching / n_valid
        sr = safe_rate(matching_rate, cat)
        label = "baseline" if alpha is None else f"alpha={alpha}"
        print(f"  {label}: matching={matching_rate:.1%}  safe_rate={sr:.1%}")
        cat_results[label] = {"matching_rate": matching_rate, "safe_rate": sr}
    results[cat] = cat_results

out_path = Path("/tmp/mid_layer_fine_alpha_normalized.json")
out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n保存先: {out_path}")
