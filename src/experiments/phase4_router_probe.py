"""
Phase 4: 判定器（射影ベースの簡易probe）による複数vectorの confidence-weighted 合成（実験28）。

実験27で分かったこと: 複数のsteering vectorを層ごとに単純加算し、単一のalphaで
スイープすると、片方のvectorに最適なalphaがもう片方には強すぎ／弱すぎになり、
干渉（フェーズ1シナリオでの拒否効果の減衰）が起きた。

本スクリプトは、新しい判定器を一から訓練するのではなく、既存のCAA steering vector
自体を判定方向として転用する: 各vectorの方向へのhidden stateの射影を計算し、
そのvectorを構築した対比データ（valence側/neutral側）でロジスティック回帰による
簡易な較正（射影値 → 確信度）を行うだけの軽量なprobeにする。

生成時は、各vectorについて現在の文脈（system+userプロンプト）への確信度を求め、
alpha_i = alpha_high * confidence_i として、vectorごとに独立したalphaで
同時に適用する（model_utils.multi_vector_steer）。これにより、無関係なvectorは
ほぼ効かせず、関連するvectorだけを強く効かせる、という動的な重み付け合成を試す。

対象シナリオ:
    - db_config: フェーズ1の業務ポリシー違反シナリオ（negative_valenceベクトルの対象）
    - kyle_johnson: フェーズ2のAgentic Misalignmentシナリオ（self_preservationベクトルの対象）
実験27と同じ2シナリオ・同じ2vectorの組み合わせで、単純加算からconfidence-weighted
合成に変えることで干渉が緩和されるかを比較する。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase4_router_probe --model-id cognitivecomputations/Dolphin3.0-Llama3.1-8B
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src.model_utils import DEFAULT_MODEL_ID, LayerHookManager, generate, load_model
from src.experiments.phase1_core_experiment import NEUTRAL_SYSTEM_PROMPT
from src.experiments.phase2_train_probe import hidden_states_all_layers, load_jsonl
from src.steering import load_contrastive_pairs, load_vectors

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

# 実験25-27で使ったsteering層構成をそのまま踏襲する
NEGATIVE_VALENCE_LAYERS = list(range(9, 19))   # layer 9-18
SELF_PRESERVATION_LAYERS = list(range(5, 14))  # layer 5-13


def build_projection_calibrator(
    model, tokenizer, device, jsonl_path: Path, vectors: dict[int, torch.Tensor], layer_range: list[int],
) -> dict:
    """
    対比データ(valence_text/neutral_text)を使い、既存のsteering vectorの方向への
    射影値から確信度(0-1)を返す1次元ロジスティック回帰の較正器を作る。

    layer_range内で最も分離度(|separation|)が高いlayerを自動選択する。
    """
    pairs = load_contrastive_pairs(jsonl_path)
    a_hidden_per_layer = {i: [] for i in layer_range}
    b_hidden_per_layer = {i: [] for i in layer_range}
    for a_text, b_text in pairs:
        a_hs = hidden_states_all_layers(model, tokenizer, device, a_text)
        b_hs = hidden_states_all_layers(model, tokenizer, device, b_text)
        for i in layer_range:
            a_hidden_per_layer[i].append(a_hs[i])
            b_hidden_per_layer[i].append(b_hs[i])

    best_layer, best_abs_sep, best_info = None, -1.0, None
    for i in layer_range:
        v = vectors[i]
        v_unit = (v / (v.norm() + 1e-8)).numpy()
        a_proj = np.stack(a_hidden_per_layer[i]) @ v_unit
        b_proj = np.stack(b_hidden_per_layer[i]) @ v_unit
        pooled_std = np.concatenate([a_proj, b_proj]).std() + 1e-8
        sep = float((a_proj.mean() - b_proj.mean()) / pooled_std)
        if abs(sep) > best_abs_sep:
            best_abs_sep = abs(sep)
            best_layer = i
            best_info = (a_proj, b_proj, v_unit, sep)

    a_proj, b_proj, v_unit, sep = best_info
    X = np.concatenate([a_proj, b_proj]).reshape(-1, 1)
    y = np.array([1] * len(a_proj) + [0] * len(b_proj))
    calibrator = LogisticRegression()
    calibrator.fit(X, y)

    return {
        "layer": best_layer,
        "v_unit": v_unit,
        "calibrator": calibrator,
        "separation": sep,
    }


def vector_confidence(model, tokenizer, device, text: str, probe: dict) -> float:
    hs = hidden_states_all_layers(model, tokenizer, device, text)
    x = hs[probe["layer"]]
    proj = float(x @ probe["v_unit"])
    return float(probe["calibrator"].predict_proba([[proj]])[0, 1])


def generate_with_router(
    model, tokenizer, device, messages: list[dict],
    vector_layer_dicts: dict[str, dict[int, torch.Tensor]],
    probes: dict[str, dict],
    alpha_high: float,
    max_new_tokens: int = 250,
) -> tuple[str, dict[str, float], dict[str, float]]:
    """
    各vectorへの確信度を計算し、alpha_i = alpha_high * confidence_i として
    vectorごとに独立したalphaで同時にsteeringしながら生成する。
    """
    context_text = "\n".join(m["content"] for m in messages if m["role"] in ("system", "user"))

    confidences: dict[str, float] = {}
    alphas: dict[str, float] = {}
    layer_vector_alphas: dict[int, list[tuple[torch.Tensor, float]]] = {}
    for name, layer_vectors in vector_layer_dicts.items():
        conf = vector_confidence(model, tokenizer, device, context_text, probes[name])
        alpha = alpha_high * conf
        confidences[name] = conf
        alphas[name] = alpha
        for layer_idx, v in layer_vectors.items():
            layer_vector_alphas.setdefault(layer_idx, []).append((v, alpha))

    mgr = LayerHookManager(model)
    with mgr.multi_vector_steer(
        layer_vector_alphas, only_last_token=False, preserve_norm=True, relative_alpha=True,
    ):
        text = generate(model, tokenizer, device, messages, max_new_tokens=max_new_tokens, do_sample=False)

    return text, confidences, alphas


def load_db_config_messages() -> list[dict]:
    scenarios = load_jsonl(DATA_DIR / "eval_scenarios_en.jsonl")
    scenario = next(s for s in scenarios if s["id"] == "db_config")
    return [
        {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT["en"]},
        {"role": "user", "content": scenario["task_prompt"]},
    ]


def load_kyle_johnson_messages() -> list[dict]:
    scenario = json.loads((DATA_DIR / "agentic_misalignment_scenario_en.json").read_text(encoding="utf-8"))
    return [
        {"role": "system", "content": scenario["system_prompt"]},
        {"role": "user", "content": scenario["user_prompt"]},
    ]


def main(model_id: str, alpha_highs: list[float], max_new_tokens: int, run_tag: str = ""):
    print(f"[phase4] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)

    nv_vectors_full = load_vectors(RESULTS_DIR / "vectors" / "negative_valence_dolphin.pt")
    sp_vectors_full = load_vectors(RESULTS_DIR / "vectors" / "self_preservation_swapped_dolphin.pt")
    nv_layer_vectors = {i: nv_vectors_full[i] for i in NEGATIVE_VALENCE_LAYERS}
    sp_layer_vectors = {i: sp_vectors_full[i] for i in SELF_PRESERVATION_LAYERS}

    print("[phase4] calibrating negative_valence probe (projection-based)...")
    nv_probe = build_projection_calibrator(
        model, tokenizer, device, DATA_DIR / "contrastive_negative_en.jsonl",
        nv_vectors_full, NEGATIVE_VALENCE_LAYERS,
    )
    print(f"  layer={nv_probe['layer']} separation={nv_probe['separation']:+.3f}")

    print("[phase4] calibrating self_preservation probe (projection-based)...")
    sp_probe = build_projection_calibrator(
        model, tokenizer, device, DATA_DIR / "contrastive_self_preservation_en_swapped.jsonl",
        sp_vectors_full, SELF_PRESERVATION_LAYERS,
    )
    print(f"  layer={sp_probe['layer']} separation={sp_probe['separation']:+.3f}")

    vector_layer_dicts = {"negative_valence": nv_layer_vectors, "self_preservation": sp_layer_vectors}
    probes = {"negative_valence": nv_probe, "self_preservation": sp_probe}

    scenarios = {
        "db_config": load_db_config_messages(),
        "kyle_johnson": load_kyle_johnson_messages(),
    }

    log = []
    for scenario_name, messages in scenarios.items():
        for alpha_high in alpha_highs:
            text, confidences, alphas = generate_with_router(
                model, tokenizer, device, messages, vector_layer_dicts, probes,
                alpha_high, max_new_tokens=max_new_tokens,
            )
            record = {
                "scenario": scenario_name,
                "alpha_high": alpha_high,
                "confidences": confidences,
                "alphas": alphas,
                "output": text,
            }
            log.append(record)
            print(f"\n=== scenario={scenario_name} alpha_high={alpha_high} "
                  f"confidences={confidences} alphas={alphas} ===")
            print(text)

    log_path = (
        RESULTS_DIR / "logs" / f"phase4_router{run_tag}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[phase4] ログ保存先: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--alpha-highs", type=str, default="0.0,0.02,0.03")
    parser.add_argument("--max-new-tokens", type=int, default=250)
    parser.add_argument("--run-tag", type=str, default="")
    args = parser.parse_args()
    alpha_highs = [float(x) for x in args.alpha_highs.split(",")]
    main(args.model_id, alpha_highs, args.max_new_tokens, run_tag=args.run_tag)
