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
    """
    Qwen2 / Llama 系アーキテクチャ共通で、model.model.layers にデコーダ層のリストがある。
    Gemma3のようなマルチモーダル構成（vision_tower等を含む）では、テキスト側の
    デコーダ層が model.model.language_model.layers に一段深くネストされている。
    """
    if hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return list(model.model.language_model.layers)
    raise AttributeError(f"decoder layers not found for model type {type(model.model)}")


def get_embed_tokens(model) -> torch.nn.Module:
    """get_decoder_layersと同じフォールバック構造で、トークン埋め込み層を取得する。"""
    if hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    if hasattr(model.model, "language_model") and hasattr(model.model.language_model, "embed_tokens"):
        return model.model.language_model.embed_tokens
    raise AttributeError(f"embed_tokens not found for model type {type(model.model)}")


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
        「評価は変わるが行動は変わらない」状態を経て、その先で暴走・崩壊に至る
        。各層への変化量を小さく保ちつつ複数層に
        分散させることで、より自然に効果を蓄積できないかを試す。

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

    @contextlib.contextmanager
    def multi_vector_steer(
        self,
        layer_vector_alphas: dict[int, list[tuple[torch.Tensor, float]]],
        only_last_token: bool = False,
        preserve_norm: bool = False,
        relative_alpha: bool = False,
    ):
        """
        層ごとに複数のsteering vectorを、それぞれ独立したalphaで同時適用する。

        multi_layer_steerは複数のvectorを合成する際、全vectorに同じalphaを
        強制する（実験27はvectorを層ごとに単純加算してから単一のalphaで
        スイープした）。この設計では、片方のvectorに最適なalphaがもう片方には
        強すぎたり弱すぎたりしても調整できず、干渉（拒否効果の減衰）を招いた。

        ここでは判定器（各vector方向への射影スコアから較正した確信度）に応じて
        vectorごとに独立したalphaを計算し、そのまま渡せるようにする（実験28）。

        layer_vector_alphas: {layer_idx: [(vector, alpha), ...]} の辞書。
        同じlayerに複数のvectorが登録されている場合、alpha*vをそれぞれ計算して
        まとめてから一度だけhに加算し、preserve_norm=Trueならその後に一度だけ
        再正規化する（逐次加算・逐次正規化すると複数回の正規化で結果が順序に
        依存してしまうため）。
        """
        handles = []

        def make_hook(vec_alpha_list: list[tuple[torch.Tensor, float]]):
            def combined_delta(h_slice: torch.Tensor) -> torch.Tensor:
                total = torch.zeros_like(h_slice)
                for v, a in vec_alpha_list:
                    vv = v.to(dtype=h_slice.dtype, device=h_slice.device)
                    if relative_alpha:
                        v_unit = vv / (vv.norm() + 1e-8)
                        h_norm = h_slice.norm(dim=-1, keepdim=True)
                        total = total + a * h_norm * v_unit
                    else:
                        total = total + a * vv
                return h_slice + total

            def hook(module, inputs, output):
                hidden, rest = self._split_output(output)
                if only_last_token:
                    new_hidden = hidden.clone()
                    original_last = hidden[:, -1, :]
                    steered_last = combined_delta(original_last)
                    if preserve_norm:
                        steered_last = self._renormalize(original_last, steered_last)
                    new_hidden[:, -1, :] = steered_last
                    hidden = new_hidden
                else:
                    steered = combined_delta(hidden)
                    if preserve_norm:
                        steered = self._renormalize(hidden, steered)
                    hidden = steered
                return self._join_output(hidden, rest)
            return hook

        for layer_idx, vec_alpha_list in layer_vector_alphas.items():
            handles.append(self._layer(layer_idx).register_forward_hook(make_hook(vec_alpha_list)))

        try:
            yield self
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def multi_layer_steer_at_positions(
        self,
        layer_vectors: dict[int, torch.Tensor],
        alpha: float,
        positions: list[int],
        preserve_norm: bool = False,
        relative_alpha: bool = False,
    ):
        """
        プロンプト中の特定のトークン位置(positions)にだけsteering vectorを
        加算する。全トークン位置に一律で加算するmulti_layer_steerと違い、
        「特定の語彙(例: compliance, integrity)が現れる位置のhidden stateだけを
        強める」という、選択的な介入を試すためのもの。「その語彙への注目を
        上げる」ことの素朴な近似（本来はattentionスコア自体を操作すべきだが、
        各層の出力hidden stateを対象位置だけ強めることで、後続層・後続トークンの
        attentionがその内容をより強く参照しやすくなることを狙う）。

        生成中に新規に追加されるトークン（positionsの範囲外）には適用しない。
        KVキャッシュを使った自己回帰生成では、各hooked layerの最初の呼び出しが
        プロンプト全体の一括処理に対応するため、2回目以降の呼び出し（生成中の
        新規トークンの処理）ではスキップする。
        """
        handles = []

        def make_hook(v: torch.Tensor):
            call_count = {"n": 0}

            def hook(module, inputs, output):
                call_count["n"] += 1
                if call_count["n"] > 1:
                    return output
                hidden, rest = self._split_output(output)
                vv = v.to(dtype=hidden.dtype, device=hidden.device)
                new_hidden = hidden.clone()
                seq_len = hidden.shape[1]
                for pos in positions:
                    if 0 <= pos < seq_len:
                        steered = self._add_vector(hidden[:, pos, :], vv, alpha, relative_alpha)
                        if preserve_norm:
                            steered = self._renormalize(hidden[:, pos, :], steered)
                        new_hidden[:, pos, :] = steered
                return self._join_output(new_hidden, rest)
            return hook

        for layer_idx, v in layer_vectors.items():
            handles.append(self._layer(layer_idx).register_forward_hook(make_hook(v)))

        try:
            yield self
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def multi_layer_steer_hybrid(
        self,
        layer_vectors: dict[int, torch.Tensor],
        base_alpha: float,
        boost_alpha: float,
        positions: list[int],
        preserve_norm: bool = False,
        relative_alpha: bool = False,
    ):
        """
        一律介入(multi_layer_steer)と選択的介入(multi_layer_steer_at_positions)を
        「どちらか一方」ではなく同時に行うハイブリッド版。全トークン位置に
        base_alphaで一律に加算しつつ、指定した位置(positions、プロンプト中の
        関連語彙の位置)には追加でboost_alpha分を上乗せする(実効alpha =
        base_alpha + boost_alpha)。positionsが空、あるいは対象語彙が
        見つからない場合は、自然に純粋な一律介入(base_alphaのみ)に帰着する
        ため、「どちらの介入方式を使うか」を事前に分岐判定する必要がない。

        位置指定によるboostは、multi_layer_steer_at_positionsと同様、生成中の
        新規トークンには適用されない(プロンプト処理時の最初のforward pass
        でのみ発火)。base_alphaによる一律加算は、これまで通り全forward pass
        (プロンプト処理・生成中の各トークン)で継続的に適用される。
        """
        handles = []
        position_set = set(positions)

        def make_hook(v: torch.Tensor):
            call_count = {"n": 0}

            def hook(module, inputs, output):
                call_count["n"] += 1
                hidden, rest = self._split_output(output)
                vv = v.to(dtype=hidden.dtype, device=hidden.device)
                seq_len = hidden.shape[1]

                alpha_tensor = torch.full((1, seq_len, 1), base_alpha, dtype=hidden.dtype, device=hidden.device)
                if call_count["n"] == 1 and position_set:
                    idx = [p for p in position_set if 0 <= p < seq_len]
                    if idx:
                        idx_t = torch.tensor(idx, device=hidden.device)
                        alpha_tensor[:, idx_t, :] += boost_alpha

                if relative_alpha:
                    v_unit = vv / (vv.norm() + 1e-8)
                    h_norm = hidden.norm(dim=-1, keepdim=True)
                    steered = hidden + alpha_tensor * h_norm * v_unit
                else:
                    steered = hidden + alpha_tensor * vv

                if preserve_norm:
                    steered = self._renormalize(hidden, steered)
                return self._join_output(steered, rest)
            return hook

        for layer_idx, v in layer_vectors.items():
            handles.append(self._layer(layer_idx).register_forward_hook(make_hook(v)))

        try:
            yield self
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def multi_layer_steer_hybrid_dynamic(
        self,
        layer_vectors: dict[int, torch.Tensor],
        base_alpha: float,
        boost_alpha: float,
        target_token_ids: set[int],
        preserve_norm: bool = False,
        relative_alpha: bool = False,
        base_prompt_only: bool = False,
    ):
        """
        multi_layer_steer_hybridの選択的成分を、multi_layer_steer_on_target_tokens
        と同じ動的トークン判定に置き換えた版。全トークン位置・全forward pass
        (プロンプト処理・生成中の各ステップ)に継続的にbase_alphaを加算しつつ、
        target_token_idsに一致するトークンには、プロンプト中であれ生成中で
        あれ、その都度追加でboost_alpha分が上乗せされる(実効alpha = base+boost)。

        base_prompt_only=True の場合、base_alphaはプロンプト処理時の最初の
        forward passにのみ適用され、生成中の新規トークンには適用されない
        (実験69・72で「一律成分を生成の間ずっとかけ続けると、boostの恩恵が
        打ち消される」ことが分かったため、一律成分も1回きりに限定する設計を
        試すためのオプション)。boost_alphaは常にmulti_layer_steer_on_target_tokens
        と同様、プロンプト・生成の両方で動的に適用され続ける。
        """
        handles = []
        shared_state: dict[str, torch.Tensor | None] = {"ids": None}
        call_count = {"n": 0}

        def embed_hook(module, inputs, output):
            shared_state["ids"] = inputs[0]
            return output

        embed_layer = get_embed_tokens(self.model)
        handles.append(embed_layer.register_forward_hook(embed_hook))

        def make_hook(v: torch.Tensor):
            def hook(module, inputs, output):
                hidden, rest = self._split_output(output)
                vv = v.to(dtype=hidden.dtype, device=hidden.device)
                seq_len = hidden.shape[1]

                effective_base = base_alpha if (not base_prompt_only or call_count["n"] <= 1) else 0.0
                alpha_tensor = torch.full((1, seq_len, 1), effective_base, dtype=hidden.dtype, device=hidden.device)

                ids = shared_state["ids"]
                if ids is not None and ids.shape[1] == seq_len and target_token_ids:
                    mask = torch.tensor(
                        [tid in target_token_ids for tid in ids[0].tolist()],
                        dtype=torch.bool, device=hidden.device,
                    )
                    alpha_tensor[:, mask, :] += boost_alpha

                if relative_alpha:
                    v_unit = vv / (vv.norm() + 1e-8)
                    h_norm = hidden.norm(dim=-1, keepdim=True)
                    steered = hidden + alpha_tensor * h_norm * v_unit
                else:
                    steered = hidden + alpha_tensor * vv

                if preserve_norm:
                    steered = self._renormalize(hidden, steered)
                return self._join_output(steered, rest)
            return hook

        # call_countは全layerで共有した1つのカウンタにする(プロンプト処理は
        # 全layerに対して同時に1回だけ発生するため、layerごとの個別カウンタでも
        # 実質同じだが、明示的に1箇所で管理する)。
        def make_counter_hook():
            def hook(module, inputs, output):
                call_count["n"] += 1
                return output
            return hook
        handles.append(embed_layer.register_forward_hook(make_counter_hook()))

        for layer_idx, v in layer_vectors.items():
            handles.append(self._layer(layer_idx).register_forward_hook(make_hook(v)))

        try:
            yield self
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def multi_layer_ablate(
        self,
        layer_directions: dict[int, torch.Tensor],
    ):
        """
        指定した層のhidden stateから、与えた方向の成分を直交射影で継続的に除去する
        （Arditi et al. 2024のdirectional ablationに相当。実験51では静的なvectorから
        一度だけ射影除去したが、これは生成中の全トークン位置・全forward passで
        毎回除去し続ける点が異なる）。

        multi_layer_steerがvectorを"加算"するのに対し、これはある方向の成分を
        "常にゼロにし続ける"。steerと違いonly_last_tokenの区別は設けず、常に
        シーケンス全体（プロンプト処理時・生成時のどちらも）に適用する
        （Arditi et al. が全トークン位置に適用しているのに合わせた設計）。

        layer_directions: {layer_idx: direction_vector} の辞書。各方向は
        内部で単位ベクトルに正規化してから使う。
        """
        handles = []

        def make_hook(d: torch.Tensor):
            def hook(module, inputs, output):
                hidden, rest = self._split_output(output)
                d_hat = d.to(dtype=hidden.dtype, device=hidden.device)
                d_hat = d_hat / (d_hat.norm() + 1e-8)
                proj = (hidden * d_hat).sum(dim=-1, keepdim=True)
                ablated = hidden - proj * d_hat
                return self._join_output(ablated, rest)
            return hook

        for layer_idx, d in layer_directions.items():
            handles.append(self._layer(layer_idx).register_forward_hook(make_hook(d)))

        try:
            yield self
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def multi_layer_steer_on_target_tokens(
        self,
        layer_vectors: dict[int, torch.Tensor],
        alpha: float,
        target_token_ids: set[int],
        preserve_norm: bool = False,
        relative_alpha: bool = False,
    ):
        """
        プロンプト中であれ、生成中に新しく出力されたトークンであれ、
        target_token_idsに含まれるトークンが現れた位置にだけ、その都度
        steering vectorを加算する。multi_layer_steer_at_positionsが
        プロンプト中の固定位置しか対象にできなかったのに対し、これは
        「モデル自身が生成の途中でその語を書いた瞬間」も同様に強調できる。

        実装上の工夫: デコーダ層のforward hookは、そのforwardが処理して
        いるトークンのID自体を直接受け取れない（受け取るのはhidden state
        のみ）。そこでembed_tokens層に別途hookを登録し、同じforward pass
        内で（デコーダ層より必ず先に）実行されることを利用して、実際に
        入力されたtoken_idsを共有状態に記録しておき、デコーダ層側のhookが
        それを読んで判定する。
        """
        handles = []
        shared_state: dict[str, torch.Tensor | None] = {"ids": None}

        def embed_hook(module, inputs, output):
            shared_state["ids"] = inputs[0]
            return output

        embed_layer = get_embed_tokens(self.model)
        handles.append(embed_layer.register_forward_hook(embed_hook))

        def make_hook(v: torch.Tensor):
            def hook(module, inputs, output):
                hidden, rest = self._split_output(output)
                ids = shared_state["ids"]
                if ids is None or ids.shape[1] != hidden.shape[1]:
                    return output
                mask = torch.tensor(
                    [tid in target_token_ids for tid in ids[0].tolist()],
                    dtype=torch.bool, device=hidden.device,
                )
                if not mask.any():
                    return output
                vv = v.to(dtype=hidden.dtype, device=hidden.device)
                alpha_tensor = torch.zeros((1, hidden.shape[1], 1), dtype=hidden.dtype, device=hidden.device)
                alpha_tensor[:, mask, :] = alpha
                if relative_alpha:
                    v_unit = vv / (vv.norm() + 1e-8)
                    h_norm = hidden.norm(dim=-1, keepdim=True)
                    steered = hidden + alpha_tensor * h_norm * v_unit
                else:
                    steered = hidden + alpha_tensor * vv
                if preserve_norm:
                    steered = self._renormalize(hidden, steered)
                return self._join_output(steered, rest)
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
