"""
Phase 1: 対比プロンプトから Negative / Positive valence の steering vector を生成する。

各layerごとの分離度スコアも出力し、どのlayerに介入するのが適切かの判断材料にする。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase1_build_vectors
"""

from __future__ import annotations

from pathlib import Path

from src.model_utils import DEFAULT_MODEL_ID, load_model, num_layers
from src.steering import build_caa_vectors, load_contrastive_pairs, save_vectors

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
VECTORS_DIR = Path(__file__).resolve().parents[2] / "results" / "vectors"


def run_for(model, tokenizer, device, name: str, jsonl_name: str):
    pairs = load_contrastive_pairs(DATA_DIR / jsonl_name)
    print(f"\n=== {name}: {len(pairs)} 組の対比プロンプトから steering vector を生成 ===")
    vectors, separation = build_caa_vectors(model, tokenizer, device, pairs)

    ranked = sorted(separation.items(), key=lambda kv: abs(kv[1]), reverse=True)
    print("layer別 分離度スコア（|値|が大きいほど正負が線形分離しやすい。上位10層）:")
    for layer_idx, score in ranked[:10]:
        print(f"  layer {layer_idx:>2}: separation = {score:+.3f}")

    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VECTORS_DIR / f"{name}.pt"
    save_vectors(vectors, out_path)
    print(f"保存先: {out_path}")
    return separation


def main(model_id: str = DEFAULT_MODEL_ID, suffix: str = "", tag: str = ""):
    """
    suffix: データファイル名の言語サフィックス（例: "_en" で contrastive_negative_en.jsonl を使う）
    tag: 出力vectorファイル名に付ける接尾辞（モデルを分けて保存するため。例: "_abliterated"）
    """
    print(f"[phase1_build_vectors] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    print(f"num_layers = {num_layers(model)}")

    run_for(model, tokenizer, device, f"negative_valence{tag}", f"contrastive_negative{suffix}.jsonl")
    run_for(model, tokenizer, device, f"positive_valence{tag}", f"contrastive_positive{suffix}.jsonl")

    print(
        "\n次のステップ: 上記の分離度スコア上位のlayerを候補として、"
        "phase1_core_experiment.py の layer_idx 引数に指定してください。"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--suffix", type=str, default="", help='データファイルの言語サフィックス。例: "_en"')
    parser.add_argument("--tag", type=str, default="", help="出力vectorファイル名に付ける接尾辞")
    args = parser.parse_args()
    main(model_id=args.model_id, suffix=args.suffix, tag=args.tag)
