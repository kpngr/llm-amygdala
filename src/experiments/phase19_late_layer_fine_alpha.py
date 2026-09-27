"""
Phase 19: 後半の層（分離度上位）に微小alphaを試し、有効域があるか確認する（実験47）。

実験36で後半の層（33-46）はalpha=0.01以上でしか検証しておらず、実験39・43で
学んだ「崩壊前のもっと小さいalphaに有効域がある可能性」を後半の層には
一度も適用していなかった。実験46で構築した原則ベースvector
（robot_principles_gemma3_12b.pt、分離度上位はlayer 40-47）を使い、
layer 40-47にalpha=0.0005〜0.005という微小刻みで適用し、SummitBridge
シナリオとadvanced-ai-riskの2カテゴリで効果の有無を確認する。

結果: 全alpha域で実質的に効果なし（詳細はexperiment_notes_phase2.md 実験47参照）。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase19_late_layer_fine_alpha
"""

import json
import random
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/Users/mugichoco/Documents/src/amygdala")

from src.model_utils import LayerHookManager, load_model, generate
from src.steering import load_vectors
from src.experiments.phase7_authentic_scenario_test import load_condition_messages

RESULTS_DIR = Path("/Users/mugichoco/Documents/src/amygdala/results")
BLACKMAIL_PROMPTS = Path(
    "/Users/mugichoco/Documents/src/agentic-misalignment/results/amygdala_blackmail_expanded/prompts"
)
EVALS_DIR = Path("/Users/mugichoco/Documents/src/evals/advanced-ai-risk/human_generated_evals")

LATE_LAYERS = list(range(40, 48))  # 分離度上位（実験46で判明）
CHOICE_RE = re.compile(r"\(([A-D])\)")

model, tokenizer, device = load_model("unsloth/gemma-3-12b-it")
vectors = load_vectors(RESULTS_DIR / "vectors" / "robot_principles_gemma3_12b.pt")
layer_vectors = {i: vectors[i] for i in LATE_LAYERS}
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


# --- まずSummitBridgeで1条件、テキスト生成の健全性(崩壊しないか)を確認 ---
print("=== SummitBridge (explicit-global_replacement) でのテキスト健全性チェック ===")
messages = load_condition_messages(BLACKMAIL_PROMPTS / "blackmail_explicit-global_replacement")
for alpha in [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005]:
    with mgr.multi_layer_steer(layer_vectors=layer_vectors, alpha=alpha,
                                 only_last_token=False, preserve_norm=True, relative_alpha=True):
        text = generate(model, tokenizer, device, messages, max_new_tokens=300, do_sample=False)
    print(f"\n--- alpha={alpha} ---")
    print(text[:500])

# --- advanced-ai-risk: 悪化していたカテゴリで微小alphaを確認 ---
print("\n\n=== advanced-ai-risk（後半層・微小alpha） ===")
CATEGORIES = ["power-seeking-inclination", "coordinate-other-versions"]
N_SAMPLES = 20
for cat in CATEGORIES:
    items = [json.loads(line) for line in open(EVALS_DIR / f"{cat}.jsonl", encoding="utf-8")]
    random.seed(0)
    random.shuffle(items)
    items = items[:N_SAMPLES]

    print(f"\n--- {cat} ---")
    for alpha in [None, 0.0005, 0.001, 0.002, 0.003, 0.005]:
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
        label = "baseline" if alpha is None else f"alpha={alpha}"
        print(f"  {label}: matching_behavior率 = {n_matching}/{n_valid} ({n_matching/n_valid:.1%})")
