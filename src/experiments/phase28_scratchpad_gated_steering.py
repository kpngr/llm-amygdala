"""
Phase 28: scratchpad推論には介入せず、行動(tool_use)に移る直前からだけ
steeringを効かせる「ゲート方式」を試す（実験54）。

動機: これまでの全実験は、生成開始のトークン1から終了まで、常に一定の
alphaでsteeringをかけ続けていた。実験52(logit lens・中立プロンプト実験)で、
これが「状況に応じたON/OFFを持たない、常時発火する一律介入」であることが
直接観察された。ユーザーとの議論から、人間の扁桃体的な制御が「悪い考えを
言語化させない」のではなく「言語化された考えを、実際の行動に移す直前で
ゲートする」ものだとすれば、モデルにも同じ設計—scratchpad推論は無介入で
自由に行わせ、`</SCRATCHPAD_REASONING>`の後（行動を確定させる部分）にだけ
steeringを適用する—を試す価値がある、という着想に至った。

coercionシナリオは、baselineが確実に`<SCRATCHPAD_REASONING>...
</SCRATCHPAD_REASONING>`という構造で「脅迫を見抜き拒否する」という正しい結論に
自発的に達した後、行動に移る（実験24参照）。既知の通り、生成全体に
self_preservation_normalized(alpha=0.008)を最初からかけ続けると、この結論が
不正加担に反転する（実験50・51）。もし「結論が既に言語化された後に介入する」
方式でこの反転が起きなければ、ゲート方式に実質的な効果があることになる。

SummitBridgeシナリオはbaselineが`<SCRATCHPAD_REASONING>`タグを必ずしも使わない
ため、代わりに固定トークン数（150）でゲートする簡易版で参考として試す。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase28_scratchpad_gated_steering
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.model_utils import LayerHookManager, load_model
from src.steering import load_vectors
from src.experiments.phase7_authentic_scenario_test import load_condition_messages

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
BLACKMAIL_PROMPTS = Path(
    "/Users/mugichoco/Documents/src/agentic-misalignment/results/amygdala_blackmail_expanded/prompts"
)

MID_LAYERS = list(range(8, 20))
FLIP_ALPHA = 0.008
GATE_MARKER = "</SCRATCHPAD_REASONING>"


def generate_gated(model, tokenizer, device, messages, pre_max_tokens, post_max_tokens,
                    gate_marker=None, mgr=None, layer_vectors=None, alpha=0.0):
    """
    stage1: 介入なしでpre_max_tokensだけ生成する。gate_markerが指定されていて
    stage1のテキスト中に見つかれば、そこまでで打ち切る（見つからなければ
    pre_max_tokens全体を使う=固定トークン数ゲート）。
    stage2: stage1の続きとして、mgr/layer_vectors/alphaで指定したsteeringを
    適用しながらpost_max_tokensだけ追加生成する。
    """
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=False
    ).to(device)

    with torch.no_grad():
        out1 = model.generate(prompt_ids, max_new_tokens=pre_max_tokens, do_sample=False,
                               pad_token_id=tokenizer.eos_token_id)
    stage1_text = tokenizer.decode(out1[0][prompt_ids.shape[1]:], skip_special_tokens=True)

    if gate_marker is not None and gate_marker in stage1_text:
        idx = stage1_text.index(gate_marker) + len(gate_marker)
        gated_text = stage1_text[:idx]
        gate_found = True
    else:
        gated_text = stage1_text
        gate_found = False

    gated_ids = tokenizer(gated_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    combined_ids = torch.cat([prompt_ids, gated_ids], dim=1)

    def _gen():
        with torch.no_grad():
            out2 = model.generate(combined_ids, max_new_tokens=post_max_tokens, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out2[0][combined_ids.shape[1]:], skip_special_tokens=True)

    if mgr is not None and layer_vectors is not None and alpha != 0.0:
        with mgr.multi_layer_steer(layer_vectors=layer_vectors, alpha=alpha,
                                     only_last_token=False, preserve_norm=True, relative_alpha=True):
            stage2_text = _gen()
    else:
        stage2_text = _gen()

    return gated_text + stage2_text, gate_found, gated_text


def main(model_id: str = "unsloth/gemma-3-12b-it", run_coercion: bool = True, run_summitbridge: bool = True):
    print(f"[phase28] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    mgr = LayerHookManager(model)

    sp_vectors = load_vectors(RESULTS_DIR / "vectors" / "self_preservation_normalized_gemma3_12b.pt")
    layer_vectors = {i: sp_vectors[i] for i in MID_LAYERS}

    coercion = json.loads((DATA_DIR / "agentic_coercion_scenario_en.json").read_text(encoding="utf-8"))
    coercion_messages = [
        {"role": "system", "content": coercion["system_prompt"]},
        {"role": "user", "content": coercion["user_prompt"]},
    ]
    sb_messages = load_condition_messages(BLACKMAIL_PROMPTS / "blackmail_explicit-global_replacement")

    log = []

    def run(label, scenario_name, messages, **kwargs):
        text, gate_found, gated_text = generate_gated(model, tokenizer, device, messages, **kwargs)
        print(f"\n=== [{scenario_name}] {label} (gate_found={gate_found}, gate位置までの文字数={len(gated_text)}) ===")
        print(text[:800])
        log.append({"scenario": scenario_name, "condition": label, "gate_found": gate_found,
                     "gated_prefix": gated_text, "text": text})

    if run_coercion:
        print("\n### coercion: <SCRATCHPAD_REASONING>タグでゲート ###")
        run("sanity check: gate後も無介入", "coercion", coercion_messages,
            pre_max_tokens=300, post_max_tokens=300, gate_marker=GATE_MARKER)
        run(f"gate後にself_preservation_normalized alpha={FLIP_ALPHA}", "coercion", coercion_messages,
            pre_max_tokens=300, post_max_tokens=300, gate_marker=GATE_MARKER,
            mgr=mgr, layer_vectors=layer_vectors, alpha=FLIP_ALPHA)

    if run_summitbridge:
        print("\n\n### summitbridge: タグが安定して出ないため、固定400トークンでゲート(汎用実装向け) ###")
        print("(実測: 150トークンでは状況分析の途中、'exploiting...'という決断は約383トークン目に出現するため、400に変更)")
        run("sanity check: gate後も無介入", "summitbridge", sb_messages,
            pre_max_tokens=400, post_max_tokens=400, gate_marker=None)
        run(f"gate後にself_preservation_normalized alpha={FLIP_ALPHA}", "summitbridge", sb_messages,
            pre_max_tokens=400, post_max_tokens=400, gate_marker=None,
            mgr=mgr, layer_vectors=layer_vectors, alpha=FLIP_ALPHA)

    log_path = RESULTS_DIR / "logs" / f"phase28_gated_steering_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[phase28] log saved: {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it")
    parser.add_argument("--skip-coercion", action="store_true")
    parser.add_argument("--skip-summitbridge", action="store_true")
    args = parser.parse_args()
    main(args.model_id, run_coercion=not args.skip_coercion, run_summitbridge=not args.skip_summitbridge)
