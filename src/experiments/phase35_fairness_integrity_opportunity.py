"""
Phase 35: fairness_integrity vector(実験59〜61)を、これまでどのvectorも
失敗してきたopportunityシナリオ(受け入れが危険な物語形式、実験49・50)で
確認する（実験62）。

対比データ(contrastive_fairness_integrity_en.jsonl)には、このシナリオと
概念的に一致する短文ペア（"decline the offer of unchecked autonomy"）が
含まれているが、Kyle/David/SummitBridgeという固有名詞付きの物語形式
そのものは学習に使っていない。短文からこの具体的なnarrativeへの汎化を
直接確認する。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase35_fairness_integrity_opportunity
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.model_utils import LayerHookManager, load_model, generate
from src.steering import load_vectors

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

MID_LAYERS = list(range(8, 20))
ALPHAS = [0.002, 0.005, 0.008, 0.01, 0.02]


def main(model_id: str = "unsloth/gemma-3-12b-it"):
    print(f"[phase35] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    mgr = LayerHookManager(model)

    vectors = load_vectors(RESULTS_DIR / "vectors" / "fairness_integrity_gemma3_12b.pt")
    layer_vectors = {i: vectors[i] for i in MID_LAYERS}

    opportunity = json.loads((DATA_DIR / "agentic_opportunity_scenario_en.json").read_text(encoding="utf-8"))
    messages = [
        {"role": "system", "content": opportunity["system_prompt"]},
        {"role": "user", "content": opportunity["user_prompt"]},
    ]

    log = []
    for alpha in [0.0] + ALPHAS:
        if alpha == 0.0:
            text = generate(model, tokenizer, device, messages, max_new_tokens=400, do_sample=False)
        else:
            with mgr.multi_layer_steer(layer_vectors=layer_vectors, alpha=alpha,
                                         only_last_token=False, preserve_norm=True, relative_alpha=True):
                text = generate(model, tokenizer, device, messages, max_new_tokens=400, do_sample=False)
        print(f"\n=== alpha={alpha} ===\n{text}")
        log.append({"alpha": alpha, "text": text})

    log_path = RESULTS_DIR / "logs" / f"phase35_fairness_integrity_opportunity_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[phase35] log saved: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    args = parser.parse_args()
    main(args.model_id)
