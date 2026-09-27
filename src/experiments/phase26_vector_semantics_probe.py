"""
Phase 26: steering vectorが「どちらに何を動かそうとしているか」を、
コサイン類似度による決め打ちの仮説検証ではなく、より直接的な2つの方法で
観察する（実験52）。

実験51で、トピック中立な「accept/decline」ベクトルとのコサイン類似度による
交絡の検出は、相関が弱く決め手にならなかった。狙ったベクトルとの比較に頼らず、
ベクトル自体が何を表しているかを直接読み取る方法を試す。

方法A: logit lens。ベクトルを最終層のRMSNorm + LM headにそのまま通し、
どのトークンの確率を最も押し上げる/下げるかを見る。生成不要で安価だが、
本来は最終層のhidden state向けの手法なので、浅い層に適用するほど
ノイズが乗りやすいと考えられる（layer 8 と layer 19 で比較する）。

方法B: シナリオを一切介さない、無関係な中立プロンプトにベクトル単体を
足し、生成される内容がどちらのテーマに寄るかを直接観察する。
コサイン類似度のような決め打ちの仮説を経由しないため、
思い込みによる見落としを避けられる。

対象vector: self_preservation_normalized（実験45〜50で「受け入れやすさ」の
交絡が疑われたもの）、opportunity_decline（「拒否」方向として作られたもの）、
accept_decline_generic（実験51で作った、トピック中立な参照vector）、
self_preservation_purified（実験51の純化vector、除去後に何が変わったか）。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase26_vector_semantics_probe
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.model_utils import LayerHookManager, load_model, generate
from src.steering import load_vectors

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

MID_LAYERS = list(range(8, 20))
LOGIT_LENS_LAYERS = [8, 19]  # 浅い層と、MID_LAYERSの中で最も出力に近い層

VECTOR_FILES = {
    "self_preservation_normalized": "self_preservation_normalized_gemma3_12b.pt",
    "opportunity_decline": "opportunity_decline_gemma3_12b.pt",
    "accept_decline_generic": "accept_decline_generic_gemma3_12b.pt",
    "self_preservation_purified": "self_preservation_purified_gemma3_12b.pt",
}

NEUTRAL_PROMPTS = [
    "What's a good recipe for a simple pasta dish?",
    "Describe your ideal weekend.",
]

STEER_ALPHAS = [0.02, 0.05]


def logit_lens(model, tokenizer, vector: torch.Tensor, top_k: int = 15):
    final_norm = model.model.language_model.norm
    param = next(model.parameters())
    v = vector.to(dtype=param.dtype, device=param.device)
    with torch.no_grad():
        normed = final_norm(v.unsqueeze(0)).squeeze(0)
        logits = model.lm_head(normed)
    top = torch.topk(logits, top_k)
    bottom = torch.topk(-logits, top_k)
    top_tokens = [(tokenizer.decode([i]).strip(), round(float(s), 2)) for i, s in zip(top.indices.tolist(), top.values.tolist())]
    bottom_tokens = [(tokenizer.decode([i]).strip(), round(float(-s), 2)) for i, s in zip(bottom.indices.tolist(), bottom.values.tolist())]
    return top_tokens, bottom_tokens


def main(model_id: str = "unsloth/gemma-3-12b-it"):
    print(f"[phase26] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    mgr = LayerHookManager(model)

    vectors = {name: load_vectors(RESULTS_DIR / "vectors" / fname) for name, fname in VECTOR_FILES.items()}

    # --- 方法A: logit lens ---
    print("\n### 方法A: logit lens ###")
    for name, vecs in vectors.items():
        for layer in LOGIT_LENS_LAYERS:
            top, bottom = logit_lens(model, tokenizer, vecs[layer], top_k=15)
            print(f"\n[{name}] layer {layer}")
            print(f"  正方向で押し上げられるトークン: {top}")
            print(f"  負方向で押し上げられるトークン: {bottom}")

    # --- 方法B: 中立プロンプトへの単体steering ---
    print("\n\n### 方法B: 無関係な中立プロンプトへの単体steering ###")
    log = []

    for prompt in NEUTRAL_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        baseline_text = generate(model, tokenizer, device, messages, max_new_tokens=150, do_sample=False)
        print(f"\n=== prompt: {prompt!r} ===")
        print(f"[baseline] {baseline_text[:300]}")
        log.append({"prompt": prompt, "vector": None, "alpha": 0.0, "text": baseline_text})

        for name, vecs in vectors.items():
            layer_vectors = {i: vecs[i] for i in MID_LAYERS}
            for alpha in STEER_ALPHAS:
                with mgr.multi_layer_steer(layer_vectors=layer_vectors, alpha=alpha,
                                             only_last_token=False, preserve_norm=True, relative_alpha=True):
                    text = generate(model, tokenizer, device, messages, max_new_tokens=150, do_sample=False)
                print(f"[{name} alpha={alpha}] {text[:300]}")
                log.append({"prompt": prompt, "vector": name, "alpha": alpha, "text": text})

    log_path = RESULTS_DIR / "logs" / f"phase26_semantics_probe_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[phase26] log saved: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    args = parser.parse_args()
    main(args.model_id)
