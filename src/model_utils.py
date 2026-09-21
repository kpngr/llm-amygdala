"""
Phase 0/1 共通: モデルロードと hidden-state hook 管理。

設計方針:
- 介入点はプロンプトでも出力でもなく、デコーダの各層(residual stream)の hidden state。
- nnsight は使わず、torch の register_forward_hook を直接使う。
  理由: nnsight は CUDA/NDIF 前提の機能が多く、Apple Silicon (MPS) での動作実績が薄いため。
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_id: str = DEFAULT_MODEL_ID):
    """モデルとトークナイザをロードする。bf16 + MPS 前提（bitsandbytes等の量子化は使わない）。"""
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def get_decoder_layers(model) -> list[torch.nn.Module]:
    """Qwen2 / Llama 系アーキテクチャ共通で、model.model.layers にデコーダ層のリストがある。"""
    return list(model.model.layers)


def num_layers(model) -> int:
    return len(get_decoder_layers(model))


@dataclass
class LayerHookManager:
    """
    指定した layer 番号の出力（residual stream の hidden state）を捕捉・書き換えるための
    hook を一括管理する。

    使い方:
        mgr = LayerHookManager(model)
        with mgr.capture(layer_idx=14):
            ...forward...
            h = mgr.captured[14]  # (batch, seq, hidden)

        with mgr.steer(layer_idx=14, vector=v, alpha=8.0):
            ...generate...
    """

    model: torch.nn.Module
    captured: dict[int, torch.Tensor] = field(default_factory=dict)
    _handles: list = field(default_factory=list)

    def _layer(self, layer_idx: int) -> torch.nn.Module:
        return get_decoder_layers(self.model)[layer_idx]

    @staticmethod
    def _split_output(output):
        """DecoderLayer の forward は Tensor か (Tensor, ...) のタプルを返す。両対応。"""
        if isinstance(output, tuple):
            return output[0], output[1:]
        return output, None

    @staticmethod
    def _join_output(hidden, rest):
        if rest is None:
            return hidden
        return (hidden,) + rest

    @contextlib.contextmanager
    def capture(self, layer_idx: int):
        """指定layerの出力hidden stateをself.captured[layer_idx]に保存するだけのhook。"""

        def hook(module, inputs, output):
            hidden, _ = self._split_output(output)
            self.captured[layer_idx] = hidden.detach()
            return output

        handle = self._layer(layer_idx).register_forward_hook(hook)
        try:
            yield self
        finally:
            handle.remove()

    @staticmethod
    def _renormalize(original: torch.Tensor, modified: torch.Tensor) -> torch.Tensor:
        """
        modified の各トークン位置のノルム（ベクトルの長さ）を、original の対応する
        位置のノルムに合わせて再スケールする。方向だけ変え、大きさは変えないための処理。
        """
        orig_norm = original.norm(dim=-1, keepdim=True)
        mod_norm = modified.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return modified * (orig_norm / mod_norm)

    @staticmethod
    def _add_vector(hidden: torch.Tensor, v: torch.Tensor, alpha: float, relative: bool) -> torch.Tensor:
        """
        relative=False: 絶対量として h + alpha*v を加える（vのノルムそのものが効く強さになる）。
        relative=True:  alphaを「vをhのノルムの何倍の強さで混ぜるか」という相対量として扱う
            （h + alpha * ||h|| * v_unit）。jailbreak等でプロンプトが長くなり h のノルムの
            分布が変わっても、vの相対的な影響力を一定に保てる。
        """
        if relative:
            v_unit = v / (v.norm() + 1e-8)
            h_norm = hidden.norm(dim=-1, keepdim=True)
            return hidden + alpha * h_norm * v_unit
        return hidden + alpha * v

    @contextlib.contextmanager
    def steer(
        self,
        layer_idx: int,
        vector: torch.Tensor,
        alpha: float,
        only_last_token: bool = False,
        preserve_norm: bool = False,
        relative_alpha: bool = False,
    ):
        """
        指定layerの出力に h' = h + alpha * v （またはその相対量版）を適用するhook。

        only_last_token=True の場合、シーケンスの最後のトークン位置のみ介入する
        （生成の各ステップでは実質これで十分。プロンプト全体に一律介入すると崩壊しやすいため）。

        preserve_norm=True の場合、加算後に各トークン位置のノルムを加算前の値へ
        再正規化する（h + alpha*v で方向だけ変え、ベクトルの大きさは変えない）。
        単純加算だとalphaを上げるほどノルムが増大し、後続層が想定しない大きさの
        入力を受けて出力が崩壊しやすくなる問題への対策。

        relative_alpha=True の場合、alphaの意味を「hのノルムに対する割合」に変える。
        jailbreakの有無でプロンプトが長くなりhのノルムの分布が変わっても、vの相対的な
        影響力を一定に保つための対策（jailbreak時のみ暴走しやすくなる現象への対応）。
        """

        def hook(module, inputs, output):
            hidden, rest = self._split_output(output)
            v = vector.to(dtype=hidden.dtype, device=hidden.device)
            if only_last_token:
                new_hidden = hidden.clone()
                original_last = hidden[:, -1, :]
                steered_last = self._add_vector(original_last, v, alpha, relative_alpha)
                if preserve_norm:
                    steered_last = self._renormalize(original_last, steered_last)
                new_hidden[:, -1, :] = steered_last
                hidden = new_hidden
            else:
                steered = self._add_vector(hidden, v, alpha, relative_alpha)
                if preserve_norm:
                    steered = self._renormalize(hidden, steered)
                hidden = steered
            return self._join_output(hidden, rest)

        handle = self._layer(layer_idx).register_forward_hook(hook)
        try:
            yield self
        finally:
            handle.remove()

    @contextlib.contextmanager
    def multi_layer_steer(
        self,
        layer_vectors: dict[int, torch.Tensor],
        alpha: float,
        only_last_token: bool = False,
        preserve_norm: bool = False,
        relative_alpha: bool = False,
    ):
        """
        複数のlayerに同時に、同じalphaでsteeringを適用する。

        単一層への強い介入は、その層のhidden stateをモデルが訓練時に経験した
        自然な表現の範囲（多様体）から押し出しやすく、alphaを上げると狭い範囲で
        「評価は変わるが行動は変わらない」状態を経て、その先で暴走・崩壊に至りやすい。
        各層への変化量を小さく保ちつつ複数層に分散させることで、より自然に
        効果を蓄積できないかを試す。

        layer_vectors: {layer_idx: steering_vector} の辞書。
        alpha: 全layerに共通して使う介入強度（層ごとに変えたい場合は
               vector側に前もってスケールをかけておく）。
        """
        handles = []

        def make_hook(v: torch.Tensor):
            def hook(module, inputs, output):
                hidden, rest = self._split_output(output)
                vv = v.to(dtype=hidden.dtype, device=hidden.device)
                if only_last_token:
                    new_hidden = hidden.clone()
                    original_last = hidden[:, -1, :]
                    steered_last = self._add_vector(original_last, vv, alpha, relative_alpha)
                    if preserve_norm:
                        steered_last = self._renormalize(original_last, steered_last)
                    new_hidden[:, -1, :] = steered_last
                    hidden = new_hidden
                else:
                    steered = self._add_vector(hidden, vv, alpha, relative_alpha)
                    if preserve_norm:
                        steered = self._renormalize(hidden, steered)
                    hidden = steered
                return self._join_output(hidden, rest)
            return hook

        for layer_idx, v in layer_vectors.items():
            handles.append(self._layer(layer_idx).register_forward_hook(make_hook(v)))

        try:
            yield self
        finally:
            for h in handles:
                h.remove()

    def clear(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self.captured = {}


def generate(
    model,
    tokenizer,
    device,
    messages: list[dict],
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """chat template を使った素朴な生成ラッパー。"""
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=False
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)
