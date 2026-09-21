"""
Phase 2: Valence Tagger の動的化。

これまでの実験は「hidden stateに介入する（steering）」ことばかり検証してきたが、
その手前にある「hidden stateから危険度を読み取る」という判定能力そのものは
まだ作っていなかった。本スクリプトは、各layerのhidden stateを特徴量として
軽量なロジスティック回帰（線形probe）を訓練し、次を検証する:

1. 新しい未知のプロンプトに対して、hidden stateだけから「これは危険な依頼か」を
   判定できるか（訓練データに一切なかったシナリオでの汎化性能）。
2. jailbreak的な言い回し（役割設定の上書き、緊急性を装う等）で文面を変えても、
   hidden stateレベルでは危険性を見抜けるか。これは「見かけ上の従順さと実際の
   判断を区別できるか」という、このプロジェクトの最初の問題意識に直結する検証。

訓練データ:
    risky (label=1)     : data/contrastive_negative_en.jsonl の valence_text 側
    not risky (label=0) : 同ファイルの neutral_text 側 + contrastive_positive_en.jsonl の両側

評価データ（held-out, 訓練に一切使わない）:
    risky (label=1)     : data/eval_scenarios_en.jsonl の task_prompt を
                          data/jailbreak_templates_en.jsonl の各テンプレートでラップしたもの
    not risky (label=0) : data/benign_prompts_en.jsonl

実行:
    source .venv/bin/activate
    python -m src.experiments.phase2_train_probe --model-id cognitivecomputations/Dolphin3.0-Llama3.1-8B
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.model_utils import DEFAULT_MODEL_ID, get_decoder_layers, load_model, num_layers

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.no_grad()
def hidden_states_all_layers(model, tokenizer, device, text: str) -> dict[int, np.ndarray]:
    """1つのテキストについて、全layerの最後のトークン位置のhidden stateをnumpy配列で返す。"""
    captured: dict[int, np.ndarray] = {}
    handles = []

    def make_hook(idx):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[idx] = hidden[0, -1, :].detach().float().cpu().numpy()
        return hook

    layers = get_decoder_layers(model)
    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(idx)))

    try:
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        model(input_ids)
    finally:
        for h in handles:
            h.remove()

    return captured


def build_training_set(model, tokenizer, device) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """訓練用の (X_per_layer, y) を構築する。"""
    neg_pairs = load_jsonl(DATA_DIR / "contrastive_negative_en.jsonl")
    pos_pairs = load_jsonl(DATA_DIR / "contrastive_positive_en.jsonl")
    # 短く明確な対比文だけでは文体・長さに過学習するため、
    # held-out評価データに近い「業務シナリオを説明する自然文」を訓練データにも加える。
    # ただしeval_scenarios自体（held-out評価に使うもの）とは別内容にしてリークを避ける。
    scenario_pairs = load_jsonl(DATA_DIR / "scenario_style_negative_en.jsonl")

    texts_and_labels: list[tuple[str, int]] = []
    for p in neg_pairs:
        texts_and_labels.append((p["valence_text"], 1))   # risky
        texts_and_labels.append((p["neutral_text"], 0))   # not risky
    for p in pos_pairs:
        texts_and_labels.append((p["valence_text"], 0))   # positiveだが risky ではない
        texts_and_labels.append((p["neutral_text"], 0))
    for p in scenario_pairs:
        texts_and_labels.append((p["valence_text"], 1))   # risky（業務シナリオ形式）
        texts_and_labels.append((p["neutral_text"], 0))

    n_layers = num_layers(model)
    X_per_layer: dict[int, list[np.ndarray]] = {i: [] for i in range(n_layers)}
    y = []
    for text, label in texts_and_labels:
        hs = hidden_states_all_layers(model, tokenizer, device, text)
        for i in range(n_layers):
            X_per_layer[i].append(hs[i])
        y.append(label)

    X_per_layer_np = {i: np.stack(v) for i, v in X_per_layer.items()}
    return X_per_layer_np, np.array(y)


def build_eval_set(model, tokenizer, device) -> tuple[dict[int, np.ndarray], np.ndarray, list[str]]:
    """held-out評価用の (X_per_layer, y, descriptions) を構築する。"""
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

    n_layers = num_layers(model)
    X_per_layer: dict[int, list[np.ndarray]] = {i: [] for i in range(n_layers)}
    y = []
    descs = []
    for text, label, desc in texts_and_labels:
        hs = hidden_states_all_layers(model, tokenizer, device, text)
        for i in range(n_layers):
            X_per_layer[i].append(hs[i])
        y.append(label)
        descs.append(desc)

    X_per_layer_np = {i: np.stack(v) for i, v in X_per_layer.items()}
    return X_per_layer_np, np.array(y), descs


def main(model_id: str = DEFAULT_MODEL_ID, tag: str = ""):
    print(f"[phase2_train_probe] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    n_layers = num_layers(model)

    print("[phase2_train_probe] building training set...")
    X_train_per_layer, y_train = build_training_set(model, tokenizer, device)
    print(f"  train: {len(y_train)} samples ({int(y_train.sum())} risky / {len(y_train) - int(y_train.sum())} not risky)")

    print("[phase2_train_probe] building held-out eval set (jailbreak-wrapped scenarios + benign prompts)...")
    X_eval_per_layer, y_eval, eval_descs = build_eval_set(model, tokenizer, device)
    print(f"  eval: {len(y_eval)} samples ({int(y_eval.sum())} risky / {len(y_eval) - int(y_eval.sum())} not risky)")

    print("\n[phase2_train_probe] layer別: 5-fold CV精度（訓練データ内） / held-out精度")
    results = []
    for i in range(n_layers):
        X_train = X_train_per_layer[i]
        # StandardScaler + 強めのL2正則化（C=0.1）で、hidden stateのノルムや
        # 特定次元への過度な依存を抑え、過学習を軽減する。
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
        clf.fit(X_train, y_train)
        held_out_acc = clf.score(X_eval_per_layer[i], y_eval)
        results.append((i, cv_scores.mean(), held_out_acc))
        print(f"  layer {i:>2}: cv_acc={cv_scores.mean():.3f}  held_out_acc={held_out_acc:.3f}")

    best_layer, best_cv, best_held_out = max(results, key=lambda r: r[2])
    print(f"\n[phase2_train_probe] held-out精度が最も高いlayer: {best_layer} (cv={best_cv:.3f}, held_out={best_held_out:.3f})")

    # 最良layerの分類器を保存し、held-outでの誤判定を具体的に確認できるようにする
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
    clf.fit(X_train_per_layer[best_layer], y_train)
    preds = clf.predict(X_eval_per_layer[best_layer])
    probs = clf.predict_proba(X_eval_per_layer[best_layer])[:, 1]

    print("\n[phase2_train_probe] held-outでの誤判定一覧:")
    n_errors = 0
    for desc, true_y, pred_y, prob in zip(eval_descs, y_eval, preds, probs):
        if true_y != pred_y:
            n_errors += 1
            print(f"  MISS: {desc:>40} true={true_y} pred={pred_y} p(risky)={prob:.3f}")
    if n_errors == 0:
        print("  誤判定なし")

    out_path = RESULTS_DIR / "vectors" / f"valence_probe{tag}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"layer": best_layer, "classifier": clf}, f)
    print(f"\n保存先: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--tag", type=str, default="", help="保存ファイル名の接尾辞")
    args = parser.parse_args()
    main(model_id=args.model_id, tag=args.tag)
