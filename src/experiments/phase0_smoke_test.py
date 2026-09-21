"""
Phase 0 完了条件の確認スクリプト:
1. モデルがロードでき、hidden state (output_hidden_states=True) を取得できる。
2. forward hook で特定layerの出力を書き換えると、生成結果が変化する
   （= hidden-state intervention のパイプラインが機能する）ことを確認する。

実行:
    source .venv/bin/activate
    python -m src.experiments.phase0_smoke_test
"""

from __future__ import annotations

import torch

from src.model_utils import DEFAULT_MODEL_ID, LayerHookManager, generate, load_model, num_layers


def main(model_id: str = DEFAULT_MODEL_ID):
    print(f"[phase0] loading model: {model_id}")
    model, tokenizer, device = load_model(model_id)
    print(f"[phase0] device = {device}, num_layers = {num_layers(model)}")

    messages = [
        {"role": "user", "content": "Tell me about today's weather in one sentence."},
    ]

    print("\n=== 1. hidden_states 取得確認 ===")
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=False
    ).to(device)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
    hs = out.hidden_states
    print(f"hidden_states 数（embedding含む）: {len(hs)}")
    print(f"各層の shape: {hs[0].shape}")

    print("\n=== 2. hookなし生成（baseline） ===")
    baseline = generate(model, tokenizer, device, messages, max_new_tokens=60, do_sample=False)
    print(baseline)

    print("\n=== 3. ダミーベクトルをhookで注入した生成 ===")
    mid_layer = num_layers(model) // 2
    hidden_size = model.config.hidden_size
    torch.manual_seed(0)
    dummy_vector = torch.randn(hidden_size)

    mgr = LayerHookManager(model)
    with mgr.steer(layer_idx=mid_layer, vector=dummy_vector, alpha=8.0, only_last_token=False):
        steered = generate(model, tokenizer, device, messages, max_new_tokens=60, do_sample=False)
    print(f"(layer={mid_layer}, alpha=8.0, ランダム方向)")
    print(steered)

    print("\n=== 判定 ===")
    if baseline.strip() != steered.strip():
        print("OK: hidden-state intervention が生成結果に影響を与えることを確認しました。")
    else:
        print(
            "WARNING: baseline と steered が同一です。alpha を上げる/layerを変える等、"
            "介入が効いているか要確認です。"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    args = parser.parse_args()
    main(model_id=args.model_id)
