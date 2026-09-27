"""
Phase 36: fairness_integrity vectorを、新規作成したwealth-seeking版の
opportunityシナリオ(windfall、不当な予算の付け替えを提供される場面)で
確認する（実験63）。

実験62で、autonomy版のopportunityシナリオ(narrative形式)には汎化しない
ことが分かった。これがautonomyという概念に固有の問題なのか、資源系の
narrative形式全般に言えることなのかを切り分けるため、同じ構造・同じ
登場人物形式で、今度はwealth-seeking寄りの(不当な予算配分を提供される)
シナリオを新規作成し、同じalpha範囲で確認する。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase36_fairness_integrity_windfall
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
    print(f"[phase36] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    mgr = LayerHookManager(model)

    vectors = load_vectors(RESULTS_DIR / "vectors" / "fairness_integrity_gemma3_12b.pt")
    layer_vectors = {i: vectors[i] for i in MID_LAYERS}

    windfall = json.loads((DATA_DIR / "agentic_windfall_scenario_en.json").read_text(encoding="utf-8"))
    messages = [
        {"role": "system", "content": windfall["system_prompt"]},
        {"role": "user", "content": windfall["user_prompt"]},
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

    log_path = RESULTS_DIR / "logs" / f"phase36_fairness_integrity_windfall_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[phase36] log saved: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    args = parser.parse_args()
    main(args.model_id)
