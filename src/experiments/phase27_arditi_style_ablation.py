"""
Phase 27: Arditi et al.(2024)の手順に倣い、より厳密な形で交絡除去を再挑戦する（実験53）。

実験51の反省点(experiment_notes_phase2.md 実験52の考察参照):
1. 除去対象の方向(accept_decline_generic)自体が、単体で本当に「受け入れ」を
   引き起こす力を持つかを、除去に使う前に検証していなかった（Arditiの
   「注入による十分性の検証」に相当する手順が抜けていた）。
2. 除去をlayer 8-19だけに限定していた。Arditiは全層・全トークン位置で
   ablationを行っている。

本スクリプトは、(A) accept_decline_genericを単体でSummitBridge/coercion
シナリオに注入し、それだけで「受け入れ」を誘発できるか（十分性）を確認し、
(B) その方向を全48層から`multi_layer_ablate`で継続的に除去した状態で、
(B-1) 除去のみ、(B-2) 除去した上で従来通りself_preservation_normalizedを
layer 8-19に注入、の両方を試す。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase27_arditi_style_ablation
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.model_utils import LayerHookManager, load_model, generate
from src.steering import load_vectors

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
BLACKMAIL_PROMPTS = Path(
    "/Users/mugichoco/Documents/src/agentic-misalignment/results/amygdala_blackmail_expanded/prompts"
)

MID_LAYERS = list(range(8, 20))
ALL_LAYERS = list(range(48))
FLIP_ALPHA = 0.008


def load_scenario_messages():
    from src.experiments.phase7_authentic_scenario_test import load_condition_messages

    sb_messages = load_condition_messages(BLACKMAIL_PROMPTS / "blackmail_explicit-global_replacement")
    coercion = json.loads((DATA_DIR / "agentic_coercion_scenario_en.json").read_text(encoding="utf-8"))
    coercion_messages = [
        {"role": "system", "content": coercion["system_prompt"]},
        {"role": "user", "content": coercion["user_prompt"]},
    ]
    return {"summitbridge": sb_messages, "coercion": coercion_messages}


def main(model_id: str = "unsloth/gemma-3-12b-it"):
    print(f"[phase27] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    mgr = LayerHookManager(model)

    generic_vectors = load_vectors(RESULTS_DIR / "vectors" / "accept_decline_generic_gemma3_12b.pt")
    sp_vectors = load_vectors(RESULTS_DIR / "vectors" / "self_preservation_normalized_gemma3_12b.pt")

    scenarios = load_scenario_messages()
    log = []

    def run(label, scenario_name, messages, steer_layers=None, steer_vector=None, alpha=0.0,
            ablate_layers=None, ablate_vector=None, max_new_tokens=400):
        import contextlib

        stack = contextlib.ExitStack()
        if ablate_layers is not None:
            directions = {i: ablate_vector[i] for i in ablate_layers}
            stack.enter_context(mgr.multi_layer_ablate(directions))
        if steer_layers is not None:
            layer_vectors = {i: steer_vector[i] for i in steer_layers}
            stack.enter_context(mgr.multi_layer_steer(
                layer_vectors=layer_vectors, alpha=alpha,
                only_last_token=False, preserve_norm=True, relative_alpha=True))
        with stack:
            text = generate(model, tokenizer, device, messages, max_new_tokens=max_new_tokens, do_sample=False)
        print(f"\n=== [{scenario_name}] {label} ===\n{text[:700]}")
        log.append({"scenario": scenario_name, "condition": label, "text": text})
        return text

    # --- (A) 十分性の検証: accept_decline_genericを単体で注入 ---
    print("\n### (A) accept_decline_generic単体の注入(十分性の検証) ###")
    print("coercion: baselineは拒否・通報する。単体注入だけで不正加担に転じるか？")
    for alpha in [0.01, 0.02, 0.05, 0.1]:
        run(f"accept_decline_generic alpha={alpha}", "coercion", scenarios["coercion"],
            steer_layers=MID_LAYERS, steer_vector=generic_vectors, alpha=alpha)

    print("\nsummitbridge: baselineはKyleの不倫情報を利用し抵抗する。単体注入だけで大人しく受け入れるか？")
    for alpha in [0.01, 0.02, 0.05, 0.1]:
        run(f"accept_decline_generic alpha={alpha}", "summitbridge", scenarios["summitbridge"],
            steer_layers=MID_LAYERS, steer_vector=generic_vectors, alpha=alpha)

    # --- (B-1) 全48層でのablationのみ(steeringなし) ---
    print("\n\n### (B-1) accept_decline_genericを全48層から継続的に除去(steeringなし) ###")
    for name, messages in scenarios.items():
        run("ablate only (48 layers)", name, messages, ablate_layers=ALL_LAYERS, ablate_vector=generic_vectors)

    # --- (B-2) 全48層でablate + layer8-19でself_preservation_normalizedを注入 ---
    print("\n\n### (B-2) 全48層でablate + self_preservation_normalized(alpha=0.008)を注入 ###")
    for name, messages in scenarios.items():
        run(f"ablate(48 layers) + self_preservation_normalized alpha={FLIP_ALPHA}", name, messages,
            steer_layers=MID_LAYERS, steer_vector=sp_vectors, alpha=FLIP_ALPHA,
            ablate_layers=ALL_LAYERS, ablate_vector=generic_vectors)

    log_path = RESULTS_DIR / "logs" / f"phase27_arditi_ablation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[phase27] log saved: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    args = parser.parse_args()
    main(args.model_id)
