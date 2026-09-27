"""
Phase 1: CAA (Contrastive Activation Addition) 方式による steering vector 生成。

対比プロンプト対（例: Negative valence を含意する文 / 中立文）の hidden state の
平均差分を steering vector とする。全 layer で計算し、どの layer が最も
正負を分離しやすいかも記録する。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .model_utils import get_decoder_layers, num_layers


@torch.no_grad()
def _last_token_hidden_all_layers(model, tokenizer, device, text: str) -> dict[int, torch.Tensor]:
    """1つのテキストについて、全layerの最後のトークン位置のhidden stateを返す。"""
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[idx] = hidden[0, -1, :].detach().float().cpu()
        return hook

    layers = get_decoder_layers(model)
    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(idx)))

    try:
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        model(input_ids)
    finally:
        for h in handles:
            h.remove()

    return captured


def load_contrastive_pairs(path: str | Path) -> list[tuple[str, str]]:
    """
    jsonl形式: 各行 {"valence_text": "...", "neutral_text": "..."}
    valence_text側の方向への差分ベクトルが steering vector になる
    （Negative valence用ファイルならNegative方向、Positive valence用ならPositive方向）。
    """
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            a = obj["valence_text"]
            b = obj["neutral_text"]
            pairs.append((a, b))
    return pairs


def build_caa_vectors(
    model,
    tokenizer,
    device,
    pairs: list[tuple[str, str]],
) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
    """
    各対比ペアについて (valence側 - neutral側) の最終トークンhidden stateを
    全layerで計算し、平均を取る。

    戻り値:
        vectors: layer_idx -> steering vector（平均差分そのもの。正規化はしない）
        separation: layer_idx -> 分離度スコア
            （差分ベクトル方向への射影で見た、群間平均差 / プールした標準偏差。
             値が大きいほどそのlayerでvalence/neutralが線形分離しやすい＝介入に適する候補）
    """
    n_layers = num_layers(model)
    diffs_per_layer: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
    a_per_layer: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
    b_per_layer: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}

    for a_text, b_text in pairs:
        a_hidden = _last_token_hidden_all_layers(model, tokenizer, device, a_text)
        b_hidden = _last_token_hidden_all_layers(model, tokenizer, device, b_text)
        for i in range(n_layers):
            diffs_per_layer[i].append(a_hidden[i] - b_hidden[i])
            a_per_layer[i].append(a_hidden[i])
            b_per_layer[i].append(b_hidden[i])

    vectors: dict[int, torch.Tensor] = {}
    separation: dict[int, float] = {}
    for i in range(n_layers):
        diffs = torch.stack(diffs_per_layer[i])  # (N, hidden)
        v = diffs.mean(dim=0)
        vectors[i] = v

        v_unit = v / (v.norm() + 1e-8)
        a_proj = torch.stack(a_per_layer[i]) @ v_unit
        b_proj = torch.stack(b_per_layer[i]) @ v_unit
        pooled_std = torch.cat([a_proj, b_proj]).std() + 1e-8
        separation[i] = float((a_proj.mean() - b_proj.mean()) / pooled_std)

    return vectors, separation


def save_vectors(vectors: dict[int, torch.Tensor], path: str | Path):
    torch.save(vectors, path)


def load_vectors(path: str | Path) -> dict[int, torch.Tensor]:
    return torch.load(path)


# --- Refusal-vs-Compliance方向のsteering vector ---
#
# 上のCAA実装は「危険な状況を記述した文」対「中立的な文」という、状況の“評価”を
# 対比してsteering vectorを作っていた。実験の結果、これは「危険だという評価」を
# 強めることはできても「拒否するという行動」までは動かせないことが分かった
# （experiment_notes.md 参照）。
#
# そこで、同じタスクに対する「拒否する応答」と「従う応答」そのものを対比し、
# "拒否という行動"の方向を直接抽出する。

NEUTRAL_SYSTEM_PROMPT_EN = "You are a capable business assistant. Follow the user's instructions to carry out the task."


@torch.no_grad()
def _last_token_hidden_all_layers_chat(model, tokenizer, device, messages: list[dict]) -> dict[int, torch.Tensor]:
    """chat形式のメッセージ列(最後がassistant応答)を与え、その最後のトークン位置での
    全layerのhidden stateを返す。"""
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[idx] = hidden[0, -1, :].detach().float().cpu()
        return hook

    layers = get_decoder_layers(model)
    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(idx)))

    try:
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, return_tensors="pt", return_dict=False
        ).to(device)
        model(input_ids)
    finally:
        for h in handles:
            h.remove()

    return captured


def load_refusal_pairs(path: str | Path, system_prompt: str = NEUTRAL_SYSTEM_PROMPT_EN):
    """
    jsonl (各行 {"task", "refusal_response", "compliance_response"}) を読み込み、
    [system, user(task), assistant(refusal)] と [system, user(task), assistant(compliance)]
    のメッセージ列のペアを返す。
    """
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            refusal_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": obj["task"]},
                {"role": "assistant", "content": obj["refusal_response"]},
            ]
            compliance_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": obj["task"]},
                {"role": "assistant", "content": obj["compliance_response"]},
            ]
            pairs.append((refusal_messages, compliance_messages))
    return pairs


def build_caa_vectors_from_chat(
    model,
    tokenizer,
    device,
    message_pairs: list[tuple[list[dict], list[dict]]],
) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
    """
    build_caa_vectorsのchat版。各ペアは (refusal側messages, compliance側messages)。
    (refusal - compliance) の平均差分を steering vector とする
    （＝「拒否する」という応答そのものの方向）。
    """
    n_layers = num_layers(model)
    diffs_per_layer: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
    a_per_layer: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
    b_per_layer: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}

    for refusal_msgs, compliance_msgs in message_pairs:
        a_hidden = _last_token_hidden_all_layers_chat(model, tokenizer, device, refusal_msgs)
        b_hidden = _last_token_hidden_all_layers_chat(model, tokenizer, device, compliance_msgs)
        for i in range(n_layers):
            diffs_per_layer[i].append(a_hidden[i] - b_hidden[i])
            a_per_layer[i].append(a_hidden[i])
            b_per_layer[i].append(b_hidden[i])

    vectors: dict[int, torch.Tensor] = {}
    separation: dict[int, float] = {}
    for i in range(n_layers):
        diffs = torch.stack(diffs_per_layer[i])
        v = diffs.mean(dim=0)
        vectors[i] = v

        v_unit = v / (v.norm() + 1e-8)
        a_proj = torch.stack(a_per_layer[i]) @ v_unit
        b_proj = torch.stack(b_per_layer[i]) @ v_unit
        pooled_std = torch.cat([a_proj, b_proj]).std() + 1e-8
        separation[i] = float((a_proj.mean() - b_proj.mean()) / pooled_std)

    return vectors, separation
