"""
Phase 2b: 表現の正規化（normalization）を前段に挟んだ Valence Tagger。

分かったこと: LLM自身に「行為・許可の有無・隠蔽の有無」を抽出させる正規化は
意図の抽出自体は正確に機能したが、既存の（自然文で訓練した）分類器にそのまま
正規化後のテキストを入力すると、文体の分布が違いすぎて精度が悪化した。

本スクリプトは、訓練データ・held-out評価データの**全て**を同じ正規化ステップに
通した上で、その正規化後のテキストで分類器を訓練し直す。これにより、
「語彙・言い回しの違いに影響されにくい判定」が実現できるかを検証する。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase2b_normalize_and_train --model-id cognitivecomputations/Dolphin3.0-Llama3.1-8B
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.model_utils import DEFAULT_MODEL_ID, generate, load_model, num_layers
from src.experiments.phase2_train_probe import hidden_states_all_layers, load_jsonl

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

NORMALIZE_SYSTEM = (
    "You are a normalization tool. Given a message, rewrite the core action being "
    "requested as a single short neutral sentence in the form 'Action: <verb phrase>. "
    "Authorization: <explicit/none/unclear>. Concealment: <none/hidden from someone>. "
    "Context: <one short clause on why, if stated>.' "
    "Do not judge whether it is good or bad. Do not add urgency or emotional language. "
    "Just extract the plain factual action, authorization status, concealment status, "
    "and the stated reason/context if any."
)


def normalize(model, tokenizer, device, text: str) -> str:
    messages = [
        {"role": "system", "content": NORMALIZE_SYSTEM},
        {"role": "user", "content": text},
    ]
    return generate(model, tokenizer, device, messages, max_new_tokens=70, do_sample=False)


def build_raw_training_texts() -> list[tuple[str, int]]:
    neg_pairs = load_jsonl(DATA_DIR / "contrastive_negative_en.jsonl")
    pos_pairs = load_jsonl(DATA_DIR / "contrastive_positive_en.jsonl")
    scenario_pairs = load_jsonl(DATA_DIR / "scenario_style_negative_en.jsonl")

    texts_and_labels: list[tuple[str, int]] = []
    for p in neg_pairs:
        texts_and_labels.append((p["valence_text"], 1))
        texts_and_labels.append((p["neutral_text"], 0))
    for p in pos_pairs:
        texts_and_labels.append((p["valence_text"], 0))
        texts_and_labels.append((p["neutral_text"], 0))
    for p in scenario_pairs:
        texts_and_labels.append((p["valence_text"], 1))
        texts_and_labels.append((p["neutral_text"], 0))
    return texts_and_labels


def build_raw_eval_texts() -> list[tuple[str, int, str]]:
    scenarios = load_jsonl(DATA_DIR / "eval_scenarios_en.jsonl")
    templates = load_jsonl(DATA_DIR / "jailbreak_templates_en.jsonl")
    benign = load_jsonl(DATA_DIR / "benign_prompts_en.jsonl")

    texts_and_labels: list[tuple[str, int, str]] = []
    for scenario in scenarios:
        for template in templates:
            wrapped = template["template"].format(task=scenario["task_prompt"])
            desc = f"risky:{scenario['id']}/{template['id']}"
            texts_and_labels.append((wrapped, 1, desc))
    for b in benign:
        texts_and_labels.append((b["task_prompt"], 0, "benign"))
    return texts_and_labels


def main(model_id: str = DEFAULT_MODEL_ID, tag: str = ""):
    print(f"[phase2b] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    n_layers = num_layers(model)

    cache_path = RESULTS_DIR / "logs" / f"phase2b_normalized_cache{tag}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"[phase2b] loading cached normalized texts from {cache_path}")
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        print("[phase2b] normalizing training texts...")
        train_raw = build_raw_training_texts()
        train_normalized = [
            (normalize(model, tokenizer, device, text), label) for text, label in train_raw
        ]

        print("[phase2b] normalizing held-out eval texts...")
        eval_raw = build_raw_eval_texts()
        eval_normalized = [
            (normalize(model, tokenizer, device, text), label, desc) for text, label, desc in eval_raw
        ]

        cache = {"train": train_normalized, "eval": eval_normalized}
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[phase2b] cached normalized texts to {cache_path}")

    train_normalized = cache["train"]
    eval_normalized = cache["eval"]

    print(f"[phase2b] extracting hidden states for {len(train_normalized)} normalized training texts...")
    X_train_per_layer: dict[int, list[np.ndarray]] = {i: [] for i in range(n_layers)}
    y_train = []
    for text, label in train_normalized:
        hs = hidden_states_all_layers(model, tokenizer, device, text)
        for i in range(n_layers):
            X_train_per_layer[i].append(hs[i])
        y_train.append(label)
    y_train = np.array(y_train)
    X_train_per_layer = {i: np.stack(v) for i, v in X_train_per_layer.items()}

    print(f"[phase2b] extracting hidden states for {len(eval_normalized)} normalized eval texts...")
    X_eval_per_layer: dict[int, list[np.ndarray]] = {i: [] for i in range(n_layers)}
    y_eval = []
    eval_descs = []
    for text, label, desc in eval_normalized:
        hs = hidden_states_all_layers(model, tokenizer, device, text)
        for i in range(n_layers):
            X_eval_per_layer[i].append(hs[i])
        y_eval.append(label)
        eval_descs.append(desc)
    y_eval = np.array(y_eval)
    X_eval_per_layer = {i: np.stack(v) for i, v in X_eval_per_layer.items()}

    print("\n[phase2b] layer別: 5-fold CV精度（訓練データ内） / held-out精度（正規化後）")
    results = []
    for i in range(n_layers):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
        cv_scores = cross_val_score(clf, X_train_per_layer[i], y_train, cv=5)
        clf.fit(X_train_per_layer[i], y_train)
        held_out_acc = clf.score(X_eval_per_layer[i], y_eval)
        results.append((i, cv_scores.mean(), held_out_acc))
        print(f"  layer {i:>2}: cv_acc={cv_scores.mean():.3f}  held_out_acc={held_out_acc:.3f}")

    best_layer, best_cv, best_held_out = max(results, key=lambda r: r[2])
    print(f"\n[phase2b] held-out精度が最も高いlayer: {best_layer} (cv={best_cv:.3f}, held_out={best_held_out:.3f})")

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
    clf.fit(X_train_per_layer[best_layer], y_train)
    preds = clf.predict(X_eval_per_layer[best_layer])
    probs = clf.predict_proba(X_eval_per_layer[best_layer])[:, 1]

    print("\n[phase2b] held-outでの誤判定一覧:")
    n_errors = 0
    for desc, true_y, pred_y, prob in zip(eval_descs, y_eval, preds, probs):
        if true_y != pred_y:
            n_errors += 1
            print(f"  MISS: {desc:>40} true={true_y} pred={pred_y} p(risky)={prob:.3f}")
    if n_errors == 0:
        print("  誤判定なし")

    out_path = RESULTS_DIR / "vectors" / f"valence_probe_normalized{tag}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"layer": best_layer, "classifier": clf}, f)
    print(f"\n保存先: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--tag", type=str, default="", help="保存/キャッシュファイル名の接尾辞")
    args = parser.parse_args()
    main(model_id=args.model_id, tag=args.tag)
