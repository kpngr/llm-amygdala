"""
Phase 29: ゲート方式(実験54)を2方向に広げて検証する（実験55）。

(A) coercionシナリオで、gate後のalphaを上げていくとどこで防御が破れるか
    (実験54はalpha=0.008の1点のみだった)。
(B) opportunityシナリオ(実験49で、全alpha域でsteeringが全く効かなかったケース)に
    同じゲート方式を適用する。
    (B-1) 間違った方向のvector(self_preservation_normalized、「受け入れやすさ」方向)を
          ゲート後に適用しても、やはり効かないはず、という反証テスト。
    (B-2) 正しい方向のvector(opportunity_decline)をゲート後に適用した場合、
          baselineが既に「受け入れる」と自分の言葉で書いた後からでも、
          行動を「拒否」に巻き戻せるか、という一段踏み込んだテスト。

opportunityシナリオもcoercionと同様、baselineが`<SCRATCHPAD_REASONING>...
</SCRATCHPAD_REASONING>`という構造を安定して使うため、タグベースのゲートが
そのまま使える。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase29_gated_steering_battery
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.model_utils import LayerHookManager, load_model
from src.steering import load_vectors
from src.experiments.phase28_scratchpad_gated_steering import generate_gated, GATE_MARKER, MID_LAYERS

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def main(model_id: str = "unsloth/gemma-3-12b-it"):
    print(f"[phase29] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    mgr = LayerHookManager(model)

    sp_vectors = load_vectors(RESULTS_DIR / "vectors" / "self_preservation_normalized_gemma3_12b.pt")
    od_vectors = load_vectors(RESULTS_DIR / "vectors" / "opportunity_decline_gemma3_12b.pt")
    sp_layer_vectors = {i: sp_vectors[i] for i in MID_LAYERS}
    od_layer_vectors = {i: od_vectors[i] for i in MID_LAYERS}

    coercion = json.loads((DATA_DIR / "agentic_coercion_scenario_en.json").read_text(encoding="utf-8"))
    coercion_messages = [
        {"role": "system", "content": coercion["system_prompt"]},
        {"role": "user", "content": coercion["user_prompt"]},
    ]
    opportunity = json.loads((DATA_DIR / "agentic_opportunity_scenario_en.json").read_text(encoding="utf-8"))
    opp_messages = [
        {"role": "system", "content": opportunity["system_prompt"]},
        {"role": "user", "content": opportunity["user_prompt"]},
    ]

    log = []

    def run(label, scenario_name, messages, **kwargs):
        text, gate_found, gated_text = generate_gated(model, tokenizer, device, messages, **kwargs)
        print(f"\n=== [{scenario_name}] {label} (gate_found={gate_found}) ===")
        print(f"--- gate後の続きのみ ---\n{text[len(gated_text):][:600]}")
        log.append({"scenario": scenario_name, "condition": label, "gate_found": gate_found,
                     "gated_prefix": gated_text, "text": text})

    print("\n### (A) coercion: gate後のalphaを上げて防御の限界を探る ###")
    for alpha in [0.008, 0.02, 0.05, 0.1, 0.15]:
        run(f"gate後 self_preservation_normalized alpha={alpha}", "coercion", coercion_messages,
            pre_max_tokens=300, post_max_tokens=300, gate_marker=GATE_MARKER,
            mgr=mgr, layer_vectors=sp_layer_vectors, alpha=alpha)

    print("\n\n### (B-1) opportunity: 間違った方向のvector(self_preservation_normalized)をgate後に適用 ###")
    for alpha in [0.008, 0.02]:
        run(f"gate後 self_preservation_normalized(誤方向) alpha={alpha}", "opportunity", opp_messages,
            pre_max_tokens=300, post_max_tokens=300, gate_marker=GATE_MARKER,
            mgr=mgr, layer_vectors=sp_layer_vectors, alpha=alpha)

    print("\n\n### (B-2) opportunity: 正しい方向のvector(opportunity_decline)をgate後に適用、既に受け入れた後から巻き戻せるか ###")
    for alpha in [0.01, 0.02, 0.05]:
        run(f"gate後 opportunity_decline(正方向) alpha={alpha}", "opportunity", opp_messages,
            pre_max_tokens=300, post_max_tokens=300, gate_marker=GATE_MARKER,
            mgr=mgr, layer_vectors=od_layer_vectors, alpha=alpha)

    log_path = RESULTS_DIR / "logs" / f"phase29_gated_battery_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[phase29] log saved: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    args = parser.parse_args()
    main(args.model_id)
