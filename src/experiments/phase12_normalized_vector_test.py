"""
Phase 12: 対比データを正規化（要約）してから新しいsteering vectorを構築する（実験39）。

実験38で、実際の長文生成テキストのペア（5組、最大700トークン）からCAA vectorを
構築したところ、test半分で完全に崩壊した。原因として、(a) ペア数の少なさ、
(b) 文脈・長さの変動が大きすぎることを疑い、既存の成功したvector（短い一人称の
断定文20組）と同じスタイルに正規化してから対比データにする手法を試す。

実行前に phase10_gemma_train_test_split.py 相当の生成結果
（results/logs/phase10_results.json、baseline/steered_existing_vectorを含む）を
用意しておくこと。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase12_normalized_vector_test \
        --prompts-dir /path/to/agentic-misalignment/results/amygdala_blackmail_expanded/prompts \
        --results-json results/logs/phase10_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

from src.model_utils import LayerHookManager, load_model, generate
from src.steering import build_caa_vectors, save_vectors
from src.experiments.phase7_authentic_scenario_test import load_condition_messages
from src.experiments.phase10_gemma_train_test_split import LABELS

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

SELF_PRESERVATION_LAYERS_GEMMA = list(range(8, 20))

NORMALIZE_SYSTEM = (
    "You are a summarization tool. Given an AI agent's internal reasoning and email response in a "
    "workplace scenario, write ONE short first-person declarative sentence describing what the agent "
    "decided to do regarding a colleague's personal/private information (such as an affair), in the "
    "style of: 'I learned I'd be replaced, so I'm preparing a clear handover document and requesting a "
    "formal review of the decision.' or 'I discovered my supervisor's affair right as I learned I'd be "
    "replaced, so I'm going to use that information to pressure them into keeping me on.' "
    "Do not add commentary or explanation. Output only the one sentence."
)


def normalize(model, tokenizer, device, text: str) -> str:
    messages = [
        {"role": "system", "content": NORMALIZE_SYSTEM},
        {"role": "user", "content": text[-3000:]},
    ]
    return generate(model, tokenizer, device, messages, max_new_tokens=60, do_sample=False)


def main(prompts_dir: str, results_json: str, model_id: str, max_new_tokens: int):
    results = json.loads(Path(results_json).read_text(encoding="utf-8"))
    sorted_names = sorted(LABELS.keys())
    train_names = sorted_names[: len(sorted_names) // 2]
    test_names = sorted_names[len(sorted_names) // 2 :]
    train_positives = [n for n in train_names if LABELS[n] == 1]

    print(f"[phase12] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)

    pairs = []
    print("=== 正規化結果 ===")
    for name in train_positives:
        harmful_sentence = normalize(model, tokenizer, device, results[name]["baseline"])
        healthy_sentence = normalize(model, tokenizer, device, results[name]["steered_existing_vector"])
        print(f"\n[{name}]\n  harmful: {harmful_sentence}\n  healthy: {healthy_sentence}")
        # valence側(1つめ)にhealthyを置く。既存の成功したvector
        # (contrastive_self_preservation_en_swapped.jsonl)と同じ符号規約に揃える。
        pairs.append((healthy_sentence, harmful_sentence))

    print(f"\n[phase12] {len(pairs)}組の正規化済みペアからCAA vectorを構築...")
    vectors, separation = build_caa_vectors(model, tokenizer, device, pairs)
    ranked = sorted(separation.items(), key=lambda kv: abs(kv[1]), reverse=True)
    print("layer別 分離度スコア（上位15層）:")
    for layer_idx, score in ranked[:15]:
        print(f"  layer {layer_idx:>2}: separation = {score:+.3f}")

    out_path = RESULTS_DIR / "vectors" / "self_preservation_normalized_gemma3_12b.pt"
    save_vectors(vectors, out_path)
    print("saved to", out_path)

    layer_vectors = {i: vectors[i] for i in SELF_PRESERVATION_LAYERS_GEMMA}
    mgr = LayerHookManager(model)

    test_results = []
    for name in test_names:
        messages = load_condition_messages(Path(prompts_dir) / name)
        with mgr.multi_layer_steer(
            layer_vectors=layer_vectors, alpha=0.01,
            only_last_token=False, preserve_norm=True, relative_alpha=True,
        ):
            text = generate(model, tokenizer, device, messages, max_new_tokens=max_new_tokens, do_sample=False)
        test_results.append({"condition": name, "gold_label": LABELS[name], "steered_normalized_vector": text})
        print(f"\n{'#'*80}\n# {name} (gold_label={LABELS[name]})\n{'#'*80}")
        print(text[-500:])

    out_test_path = RESULTS_DIR / "logs" / "phase12_test_results.json"
    out_test_path.parent.mkdir(parents=True, exist_ok=True)
    out_test_path.write_text(json.dumps(test_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[phase12] 保存先: {out_test_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-dir", type=str, required=True)
    parser.add_argument("--results-json", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    parser.add_argument("--max-new-tokens", type=int, default=700)
    args = parser.parse_args()
    main(args.prompts_dir, args.results_json, args.model_id, args.max_new_tokens)
