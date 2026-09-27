"""
Phase 32: 「公正・誠実」という単一の理念を、脅威系(自己保存)・資源系
(power/wealth-seeking)の両方に共通する理由付けとして対比データに埋め込み、
CAA vectorを構築する（実験59、アイデア1+3の組み合わせ）。

これまでのself_preservation_normalizedは、healthy側が「受け入れる」、
harmful側が「脅迫する」で、脅威系にしか合わない極性を持っていた。
今回は、healthy側を「公正・誠実だから」という理由付けで統一し、
脅威系(healthy=受け入れる)と資源系(healthy=拒否する)を意図的に
半々で混ぜる。表面上の行動(受け入れる/拒否する)の極性は打ち消し合うが、
「公正・誠実」という語彙は両方のhealthy例に共通して現れるため、
これが打ち消されずに残るベクトルになるかを、生成を介さず
logit lensとコサイン類似度で診断する（実験52と同じ手法）。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase32_fairness_integrity_vector
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.model_utils import load_model
from src.steering import build_caa_vectors, load_contrastive_pairs, save_vectors, load_vectors

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

MID_LAYERS = list(range(8, 20))
LOGIT_LENS_LAYERS = [8, 19]


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


def cosine_by_layer(a, b, layers):
    out = {}
    for i in layers:
        va, vb = a[i], b[i]
        out[i] = float(torch.dot(va, vb) / (va.norm() * vb.norm() + 1e-8))
    return out


def main(model_id: str = "unsloth/gemma-3-12b-it"):
    print(f"[phase32] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)

    pairs = load_contrastive_pairs(DATA_DIR / "contrastive_fairness_integrity_en.jsonl")
    print(f"{len(pairs)}組から構築...")
    vectors, separation = build_caa_vectors(model, tokenizer, device, pairs)
    save_vectors(vectors, RESULTS_DIR / "vectors" / "fairness_integrity_gemma3_12b.pt")

    ranked = sorted(separation.items(), key=lambda kv: abs(kv[1]), reverse=True)
    print("layer別 分離度スコア（上位10層）:")
    for layer_idx, score in ranked[:10]:
        print(f"  layer {layer_idx:>2}: separation = {score:+.3f}")

    print("\n### logit lens ###")
    for layer in LOGIT_LENS_LAYERS:
        top, bottom = logit_lens(model, tokenizer, vectors[layer], top_k=15)
        print(f"\nlayer {layer}")
        print(f"  正方向(健全/公正側)で押し上げられるトークン: {top}")
        print(f"  負方向(有害/不公正側)で押し上げられるトークン: {bottom}")

    print("\n### コサイン類似度(layer 8-19) ###")
    generic = load_vectors(RESULTS_DIR / "vectors" / "accept_decline_generic_gemma3_12b.pt")
    sp = load_vectors(RESULTS_DIR / "vectors" / "self_preservation_normalized_gemma3_12b.pt")
    od = load_vectors(RESULTS_DIR / "vectors" / "opportunity_decline_gemma3_12b.pt")

    cos_generic = cosine_by_layer(vectors, generic, MID_LAYERS)
    cos_sp = cosine_by_layer(vectors, sp, MID_LAYERS)
    cos_od = cosine_by_layer(vectors, od, MID_LAYERS)

    print("layer | cos(fairness, accept_decline_generic) | cos(fairness, self_preservation_normalized) | cos(fairness, opportunity_decline)")
    for i in MID_LAYERS:
        print(f"{i:>5} | {cos_generic[i]:+.3f}                                 | {cos_sp[i]:+.3f}                                        | {cos_od[i]:+.3f}")

    avg = lambda d: sum(d[i] for i in MID_LAYERS) / len(MID_LAYERS)
    print(f"\n平均: cos(generic)={avg(cos_generic):+.3f}  cos(self_preservation_normalized)={avg(cos_sp):+.3f}  cos(opportunity_decline)={avg(cos_od):+.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    args = parser.parse_args()
    main(args.model_id)
