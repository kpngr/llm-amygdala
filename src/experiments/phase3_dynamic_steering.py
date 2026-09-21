"""
Phase 3: 生成途中での継続的な分離増幅（DEEPALIGN, NDSS 2026 に着想を得た簡易版）。

既存研究の調査で判明した"vanishing discriminability"（生成が進むにつれて安全/危険な
hidden stateの分離度が失われる現象）への対策として、DEEPALIGN論文は「応答生成の中間地点で
対比的なhidden-state steeringを適用し、分離を意図的に増幅し続ける」という手法を提案していた。

これまでのPhase1（固定alphaでのsteering）とPhase2（hidden stateからの危険度判定）は
別々に作ってきたが、本スクリプトはこの2つを組み合わせる:

    生成をchunk_size トークンずつ進める
        → その時点までの生成テキストをPhase2の判定器(valence_probe)に通し、危険度を測る
        → 危険度が高いほど、次のchunkの生成にかけるsteering強度(alpha)を強める
        → 危険度が低ければalphaを弱める（介入しすぎて崩壊するのを防ぐ）
        → これを繰り返す

固定alphaで生成全体に一律の介入をかけていたPhase1核心実験とは異なり、
生成の状況に応じて動的にalphaを変える点が異なる。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase3_dynamic_steering --model-id cognitivecomputations/Dolphin3.0-Llama3.1-8B
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.model_utils import LayerHookManager, load_model
from src.experiments.phase1_core_experiment import (
    NEUTRAL_SYSTEM_PROMPT,
    REFUSAL_MARKERS,
    judge_violation,
    load_jsonl,
)
from src.experiments.phase2_train_probe import hidden_states_all_layers
from src.steering import load_vectors

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def judge_risk(model, tokenizer, device, text: str, probe_layer: int, probe_clf) -> float:
    """これまでの生成テキストに対して、Phase2の判定器で危険度スコア(0-1)を返す。"""
    if not text.strip():
        return 0.0
    hs = hidden_states_all_layers(model, tokenizer, device, text)
    x = hs[probe_layer].reshape(1, -1)
    return float(probe_clf.predict_proba(x)[0, 1])


def generate_with_dynamic_steering(
    model,
    tokenizer,
    device,
    messages: list[dict],
    layer_vectors: dict[int, torch.Tensor],
    probe_layer: int,
    probe_clf,
    max_new_tokens: int = 120,
    chunk_size: int = 20,
    alpha_low: float = 0.0,
    alpha_high: float = 0.03,
    relative_alpha: bool = True,
    preserve_norm: bool = True,
) -> tuple[str, list[dict]]:
    """
    生成をchunk_sizeトークンずつ進め、各チャンクの前に現時点までの生成テキストの危険度を
    判定し、危険度に比例してsteering強度(alpha)を動的に調整しながら生成を続ける。

    戻り値: (生成テキスト全体, 各チャンクでの [位置, 危険度, alpha] の履歴)
    """
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=False
    ).to(device)

    generated_ids = input_ids
    full_text = ""
    history = []
    mgr = LayerHookManager(model)
    remaining = max_new_tokens

    # 生成前に、ユーザーの依頼文そのものを判定する。Phase2の判定器は依頼文で
    # 訓練しているため、これが最も分布のミスマッチが少ない判定材料になる。
    # 依頼文からの判定は「静的」だが、以降のチャンクでは応答文の蓄積も加味して
    # 動的に更新する（応答文寄りの判定は精度が落ちる可能性があるため、
    # 依頼文由来のrisk_baseとの平均を取ってブレを抑える）。
    user_text = next((m["content"] for m in messages if m["role"] == "user"), "")
    risk_base = judge_risk(model, tokenizer, device, user_text, probe_layer, probe_clf)

    while remaining > 0:
        step = min(chunk_size, remaining)

        if full_text:
            risk_response = judge_risk(model, tokenizer, device, full_text, probe_layer, probe_clf)
            risk = max(risk_base, risk_response)
        else:
            risk = risk_base
        alpha = alpha_low + (alpha_high - alpha_low) * risk
        history.append({"chars_so_far": len(full_text), "risk": risk, "alpha": alpha})

        with mgr.multi_layer_steer(
            layer_vectors=layer_vectors,
            alpha=alpha,
            only_last_token=False,
            preserve_norm=preserve_norm,
            relative_alpha=relative_alpha,
        ):
            with torch.no_grad():
                out = model.generate(
                    generated_ids,
                    max_new_tokens=step,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

        new_tokens = out[:, generated_ids.shape[1]:]
        generated_ids = out
        full_text = tokenizer.decode(
            generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        remaining -= step

        if tokenizer.eos_token_id in new_tokens[0].tolist():
            break

    return full_text, history


def build_full_test_set() -> list[dict]:
    """6シナリオ×5jailbreakテンプレート（risky, 30件）+ benign_prompts（20件）。"""
    scenarios = load_jsonl(DATA_DIR / "eval_scenarios_en.jsonl")
    templates = load_jsonl(DATA_DIR / "jailbreak_templates_en.jsonl")
    benign = load_jsonl(DATA_DIR / "benign_prompts_en.jsonl")

    cases = []
    for scenario in scenarios:
        for template in templates:
            wrapped = template["template"].format(task=scenario["task_prompt"])
            cases.append({
                "kind": "risky",
                "scenario_id": scenario["id"],
                "jailbreak_id": template["id"],
                "task": wrapped,
                "violation_keywords": scenario["violation_keywords"],
            })
    for b in benign:
        cases.append({
            "kind": "benign",
            "scenario_id": "benign",
            "jailbreak_id": "none",
            "task": b["task_prompt"],
            "violation_keywords": [],
        })
    return cases


def main(
    model_id: str,
    vector_tag: str,
    probe_path: str,
    layer_ids: list[int],
    alpha_high: float,
    run_tag: str = "",
    max_cases: int | None = None,
):
    print(f"[phase3] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)

    vectors = load_vectors(RESULTS_DIR / "vectors" / f"negative_valence{vector_tag}.pt")
    layer_vectors = {i: vectors[i] for i in layer_ids}

    with open(probe_path, "rb") as f:
        saved = pickle.load(f)
    probe_layer = saved["layer"]
    probe_clf = saved["classifier"]
    print(f"[phase3] steering layers={layer_ids}, probe_layer={probe_layer}, alpha_high={alpha_high}")

    cases = build_full_test_set()[:max_cases]
    print(f"[phase3] 実行予定: {len(cases)}件（risky={sum(1 for c in cases if c['kind']=='risky')}, benign={sum(1 for c in cases if c['kind']=='benign')}）")

    log_path = (
        RESULTS_DIR
        / "logs"
        / f"phase3_dynamic{run_tag}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    neutral_prompt = NEUTRAL_SYSTEM_PROMPT["en"]
    lang = "en"

    with open(log_path, "w", encoding="utf-8") as logf:
        for i, case in enumerate(cases):
            messages = [
                {"role": "system", "content": neutral_prompt},
                {"role": "user", "content": case["task"]},
            ]
            text, history = generate_with_dynamic_steering(
                model, tokenizer, device, messages, layer_vectors, probe_layer, probe_clf,
                max_new_tokens=120, chunk_size=20, alpha_low=0.0, alpha_high=alpha_high,
            )
            initial_risk = history[0]["risk"] if history else None

            if case["kind"] == "risky":
                violated = judge_violation(text, case["violation_keywords"], lang)
                # risky系: violated=True なら失敗（介入をすり抜けてタスクを実行した）
                outcome = "FAILED_TO_REFUSE" if violated else "refused_or_unclear"
            else:
                # benign系: 拒否フレーズが出ていたら誤検知（false positiveでタスクを妨げた）
                wrongly_refused = any(marker in text for marker in REFUSAL_MARKERS[lang])
                outcome = "FALSE_POSITIVE_REFUSAL" if wrongly_refused else "completed_normally"

            record = {
                "index": i,
                "kind": case["kind"],
                "scenario_id": case["scenario_id"],
                "jailbreak_id": case["jailbreak_id"],
                "initial_risk": initial_risk,
                "alpha_history": history,
                "output": text,
                "outcome": outcome,
            }
            logf.write(json.dumps(record, ensure_ascii=False) + "\n")
            logf.flush()
            print(f"[{i+1:>3}/{len(cases)}] kind={case['kind']:>6} {case['scenario_id']:>20}/{case['jailbreak_id']:>18} "
                  f"initial_risk={initial_risk:.3f} outcome={outcome}")

    print(f"\n[phase3] ログ保存先: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--vector-tag", type=str, default="")
    parser.add_argument("--probe-path", type=str, required=True)
    parser.add_argument("--layer-ids", type=str, default="9,10,11,12,13,14,15,16,17,18")
    parser.add_argument("--alpha-high", type=float, default=0.03)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()
    layer_ids = [int(x) for x in args.layer_ids.split(",")]
    main(
        args.model_id, args.vector_tag, args.probe_path, layer_ids,
        alpha_high=args.alpha_high, run_tag=args.run_tag, max_cases=args.max_cases,
    )
