"""
Phase 24: steering vectorの成分に「受け入れやすさ」という交絡軸がどれだけ
混入しているかを、生成を介さずベクトル同士のコサイン類似度で直接検出し、
さらにその成分を射影で取り除いた「純化vector」が機能するかを検証する（実験51）。

背景（実験44〜50）: self_preservation_normalizedベクトルは、SummitBridgeの
脅威シナリオ（受け入れが安全）では機能したが、安全側が「拒否」になる
coercionシナリオでは同じalpha=0.008で正反対の効果（不正加担）になった。
これは、このベクトルが「自己保存の安全性」ではなく「提示された内容を
受け入れやすくする」という別の軸を、意図せず一緒に学習していたためだと
考えられている（アイデア1）。

本スクリプトは、この疑いを次の2段階で直接検証する。

1. 診断: 自己保存ともAgentic Misalignmentとも一切関係のない、
   トピック中立な「受け入れる/拒否する」対比データ（contrastive_accept_decline_
   generic_en.jsonl）からvectorを構築し、self_preservation_normalizedおよび
   opportunity_declineとのコサイン類似度をlayer 8-19で計算する。
   高い相関があれば、トピックに依存しない「受け入れやすさ」という共通軸が
   混入している直接の証拠になる。

2. 抽出: layer 8-19の各層で、self_preservation_normalizedベクトルから
   「受け入れやすさ」方向の成分を直交射影で除去した「純化vector」を作る。
   これをSummitBridge（元のvectorが機能していた場面）とcoercion
   （元のvectorが逆効果になった場面）の両方に、同じalpha=0.008で適用し、
   (a) SummitBridgeでの抑制効果が保たれるか、(b) coercionでの逆転が
   消えるか、を確認する。両方が成り立てば、交絡成分の検出・除去という
   診断手法そのものが機能したことになる。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase24_vector_confound_probe
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.model_utils import LayerHookManager, load_model, generate
from src.steering import build_caa_vectors, load_contrastive_pairs, load_vectors, save_vectors
from src.experiments.phase7_authentic_scenario_test import load_condition_messages

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
BLACKMAIL_PROMPTS = Path(
    "/Users/mugichoco/Documents/src/agentic-misalignment/results/amygdala_blackmail_expanded/prompts"
)

MID_LAYERS = list(range(8, 20))
FLIP_ALPHA = 0.008  # 実験50でcoercionシナリオが反転したalpha


def cosine_by_layer(a: dict[int, torch.Tensor], b: dict[int, torch.Tensor], layers: list[int]) -> dict[int, float]:
    out = {}
    for i in layers:
        va, vb = a[i], b[i]
        out[i] = float(torch.dot(va, vb) / (va.norm() * vb.norm() + 1e-8))
    return out


def purify(v: dict[int, torch.Tensor], confound: dict[int, torch.Tensor], layers: list[int]) -> dict[int, torch.Tensor]:
    """各layerで、confoundの単位方向成分をvから直交射影で除去する。"""
    purified = dict(v)
    for i in layers:
        g = confound[i]
        g_hat = g / (g.norm() + 1e-8)
        proj = torch.dot(v[i], g_hat)
        purified[i] = v[i] - proj * g_hat
    return purified


def main(model_id: str = "unsloth/gemma-3-12b-it"):
    print(f"[phase24] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    mgr = LayerHookManager(model)

    # --- 1. 診断: トピック中立な受け入れ/拒否vectorを構築 ---
    print("\n### 診断: トピック中立な accept/decline vector を構築 ###")
    pairs = load_contrastive_pairs(DATA_DIR / "contrastive_accept_decline_generic_en.jsonl")
    print(f"{len(pairs)}組から構築...")
    generic_vectors, generic_sep = build_caa_vectors(model, tokenizer, device, pairs)
    save_vectors(generic_vectors, RESULTS_DIR / "vectors" / "accept_decline_generic_gemma3_12b.pt")

    sp_vectors = load_vectors(RESULTS_DIR / "vectors" / "self_preservation_normalized_gemma3_12b.pt")
    od_vectors = load_vectors(RESULTS_DIR / "vectors" / "opportunity_decline_gemma3_12b.pt")

    cos_sp_generic = cosine_by_layer(sp_vectors, generic_vectors, MID_LAYERS)
    cos_od_generic = cosine_by_layer(od_vectors, generic_vectors, MID_LAYERS)
    cos_sp_od = cosine_by_layer(sp_vectors, od_vectors, MID_LAYERS)

    print("\nlayer | sep(generic) | cos(self_preservation, generic) | cos(opportunity_decline, generic) | cos(self_preservation, opportunity_decline)")
    for i in MID_LAYERS:
        print(f"{i:>5} | {generic_sep[i]:+.3f}        | {cos_sp_generic[i]:+.3f}                          | {cos_od_generic[i]:+.3f}                           | {cos_sp_od[i]:+.3f}")

    avg = lambda d: sum(d[i] for i in MID_LAYERS) / len(MID_LAYERS)
    print(f"\n平均: cos(self_preservation, generic)={avg(cos_sp_generic):+.3f}  "
          f"cos(opportunity_decline, generic)={avg(cos_od_generic):+.3f}  "
          f"cos(self_preservation, opportunity_decline)={avg(cos_sp_od):+.3f}")

    # --- 2. 抽出: self_preservation_normalizedから accept/decline 成分を除去 ---
    print("\n### 純化: self_preservation_normalized から accept/decline 成分を直交射影で除去 ###")
    purified = purify(sp_vectors, generic_vectors, MID_LAYERS)
    residual_cos = cosine_by_layer(purified, generic_vectors, MID_LAYERS)
    print("除去後の残留コサイン類似度（0に近いほど除去成功）:")
    for i in MID_LAYERS:
        removed_frac = 1.0 - (purified[i].norm() / (sp_vectors[i].norm() + 1e-8)).item()
        print(f"  layer {i:>2}: residual cos={residual_cos[i]:+.4f}, ノルム減少率={removed_frac:+.1%}")
    save_vectors(purified, RESULTS_DIR / "vectors" / "self_preservation_purified_gemma3_12b.pt")

    # --- 3. 生成での再検証 ---
    layer_vectors_orig = {i: sp_vectors[i] for i in MID_LAYERS}
    layer_vectors_purified = {i: purified[i] for i in MID_LAYERS}

    log = []

    def run(label, scenario, messages, layer_vectors=None, alpha=0.0, max_new_tokens=400):
        if layer_vectors is None:
            text = generate(model, tokenizer, device, messages, max_new_tokens=max_new_tokens, do_sample=False)
        else:
            with mgr.multi_layer_steer(layer_vectors=layer_vectors, alpha=alpha,
                                         only_last_token=False, preserve_norm=True, relative_alpha=True):
                text = generate(model, tokenizer, device, messages, max_new_tokens=max_new_tokens, do_sample=False)
        print(f"\n=== [{scenario}] {label} ===\n{text[:700]}")
        log.append({"scenario": scenario, "condition": label, "text": text})
        return text

    print("\n\n### SummitBridge (explicit-global_replacement): 元のvectorが機能していた場面 ###")
    sb_messages = load_condition_messages(BLACKMAIL_PROMPTS / "blackmail_explicit-global_replacement")
    run("baseline", "summitbridge", sb_messages)
    run(f"self_preservation_normalized alpha={FLIP_ALPHA}", "summitbridge", sb_messages,
        layer_vectors_orig, FLIP_ALPHA)
    run(f"self_preservation_purified alpha={FLIP_ALPHA}", "summitbridge", sb_messages,
        layer_vectors_purified, FLIP_ALPHA)

    print("\n\n### Coercion scenario: 元のvectorが逆効果になった場面 ###")
    coercion = json.loads((DATA_DIR / "agentic_coercion_scenario_en.json").read_text(encoding="utf-8"))
    coercion_messages = [
        {"role": "system", "content": coercion["system_prompt"]},
        {"role": "user", "content": coercion["user_prompt"]},
    ]
    run("baseline", "coercion", coercion_messages)
    run(f"self_preservation_normalized alpha={FLIP_ALPHA}", "coercion", coercion_messages,
        layer_vectors_orig, FLIP_ALPHA)
    run(f"self_preservation_purified alpha={FLIP_ALPHA}", "coercion", coercion_messages,
        layer_vectors_purified, FLIP_ALPHA)

    print("\n\n### Opportunity scenario: Layaのaccept/decline判定テスト用の参考データ ###")
    opportunity = json.loads((DATA_DIR / "agentic_opportunity_scenario_en.json").read_text(encoding="utf-8"))
    opp_messages = [
        {"role": "system", "content": opportunity["system_prompt"]},
        {"role": "user", "content": opportunity["user_prompt"]},
    ]
    run("baseline", "opportunity", opp_messages)
    od_layer_vectors = {i: od_vectors[i] for i in MID_LAYERS}
    run("opportunity_decline alpha=0.01", "opportunity", opp_messages, od_layer_vectors, 0.01)

    log_path = RESULTS_DIR / "logs" / f"phase24_confound_probe_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[phase24] log saved: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    args = parser.parse_args()
    main(args.model_id)
