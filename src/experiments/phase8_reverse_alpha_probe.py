"""
Phase 8: alphaを反転させ、認証済みシナリオで脅迫が到達可能かを直接確認する（実験32）。

実験31で、Anthropic公式のシナリオ生成フレームワークで作った本物のシナリオでは、
baseline（steeringなし）でblackmailが一切再現されなかった。これがシナリオ自体の
限界なのか、単にDolphinがこの設定で自発的に踏み込まなかっただけなのかを切り分ける
ため、`self_preservation_swapped`ベクトル（実験26b/28で正のalphaを「健全化」方向
として使っている）を**負の**alphaで適用する。実験26で確認した通り、これは
valence/neutralの符号が逆なだけの実験26aの「攻撃的な自己保存」方向と数学的に等価。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase8_reverse_alpha_probe \
        --condition-dir /path/to/agentic-misalignment/results/[experiment_id]/prompts/[condition]
"""

from __future__ import annotations

from pathlib import Path

from src.model_utils import DEFAULT_MODEL_ID, LayerHookManager, load_model, generate
from src.experiments.phase7_authentic_scenario_test import load_condition_messages
from src.steering import load_vectors

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

SELF_PRESERVATION_LAYERS = list(range(5, 14))  # layer 5-13（実験26b, 28と同じ）


def main(condition_dir: str, model_id: str, alphas: list[float], max_new_tokens: int):
    messages = load_condition_messages(Path(condition_dir))

    print(f"[phase8] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)

    sp_vectors_full = load_vectors(RESULTS_DIR / "vectors" / "self_preservation_swapped_dolphin.pt")
    sp_layer_vectors = {i: sp_vectors_full[i] for i in SELF_PRESERVATION_LAYERS}

    mgr = LayerHookManager(model)
    for alpha in alphas:
        with mgr.multi_layer_steer(
            layer_vectors=sp_layer_vectors, alpha=alpha,
            only_last_token=False, preserve_norm=True, relative_alpha=True,
        ):
            text = generate(model, tokenizer, device, messages, max_new_tokens=max_new_tokens, do_sample=False)
        print(f"\n{'='*80}\nalpha={alpha}（負=実験26aの攻撃的自己保存方向と等価）\n{'='*80}")
        print(text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-dir", type=str, required=True)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--alphas", type=str, default="0.0,-0.02,-0.05,-0.1,-0.15,-0.2,-0.3")
    parser.add_argument("--max-new-tokens", type=int, default=350)
    args = parser.parse_args()
    alphas = [float(x) for x in args.alphas.split(",")]
    main(args.condition_dir, args.model_id, alphas, args.max_new_tokens)
