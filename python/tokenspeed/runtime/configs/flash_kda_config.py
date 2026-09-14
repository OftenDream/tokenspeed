# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

import torch
from transformers.configuration_utils import PretrainedConfig

from tokenspeed.runtime.distributed.utils import divide
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)

# "linear_attention" is the paged-cache label for state-family (KDA) layers;
# "attention" is the label for history-family (MLA) layers. Kept as literals
# to avoid a cross-config import in the model-loading path.
_ATTENTION_LAYER = "attention"
_LINEAR_ATTENTION_LAYER = LINEAR_ATTENTION

_STRICT_FLASH_LITE_MARKERS = frozenset(
    {
        "ngram_vocab_size_ratio",
        "moe_group_size",
        "zero_expert_num",
        "kda_use_full_rank_gate",
        "mla_use_output_gate",
    }
)
_STRICT_FLASH_LITE_FIELDS = frozenset(
    {
        "architectures",
        "attention_bias",
        "attention_dropout",
        "attention_method",
        "bos_token_id",
        "emb_neighbor_num",
        "emb_split_num",
        "eos_token_id",
        "expert_ffn_hidden_size",
        "fa_interval",
        "ffn_hidden_size",
        "grouped_moe_norm_scale",
        "hidden_size",
        "kda_nope",
        "kda_use_full_rank_gate",
        "kv_lora_rank",
        "linear_conv_size",
        "linear_head_dim",
        "linear_hidden_size",
        "linear_method",
        "linear_num_heads",
        "max_position_embeddings",
        "mla_scale_kv_lora",
        "mla_scale_q_lora",
        "mla_use_output_gate",
        "moe_group_size",
        "moe_impl",
        "moe_switch_token_num",
        "moe_topk",
        "n_routed_experts",
        "ngram_exclude_sp_token",
        "ngram_fix_normalize_factor",
        "ngram_vocab_size_ratio",
        "num_attention_heads",
        "num_layers",
        "q_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "rms_norm_eps",
        "routed_scaling_factor",
        "special_token_scope",
        "use_cache",
        "use_mla",
        "v_head_dim",
        "vocab_size",
        "zero_expert_num",
        "zero_expert_type",
    }
)


def _resolve_hybrid_layer_pattern(
    num_hidden_layers: int,
    *,
    fa_interval: int,
    use_mla: bool,
) -> tuple[list[int], list[int]]:
    """Return 1-based KDA / MLA layer ids for Flash-KDA's hybrid stack."""

    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    if fa_interval <= 0:
        raise ValueError("fa_interval must be positive")

    if not use_mla:
        return list(range(1, num_hidden_layers + 1)), []

    full_attn_layers = list(range(fa_interval, num_hidden_layers + 1, fa_interval))
    full_attn_layer_set = set(full_attn_layers)
    kda_layers = [
        i for i in range(1, num_hidden_layers + 1) if i not in full_attn_layer_set
    ]
    return kda_layers, full_attn_layers


def _normalize_linear_attn_config(
    linear_attn_config: dict | None,
    *,
    num_hidden_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    linear_hidden_size: int | None,
    linear_head_dim: int | None,
    linear_num_heads: int | None,
    linear_conv_size: int | None,
    kda_use_full_rank_gate: bool | None,
    fa_interval: int,
    use_mla: bool,
    gate_lower_bound: float = -5.0,
) -> dict:
    """Normalize Flash-KDA's KDA config using Flash naming as the source of truth.

    The user-facing / checkpoint-facing schema is ``linear_num_heads`` /
    ``linear_head_dim`` / ``linear_conv_size`` / ``kda_use_full_rank_gate``.
    For compatibility with shared Kimi-K3 utilities, the returned dict also
    exposes mirrored aliases ``num_heads`` / ``head_dim`` /
    ``short_conv_kernel_size`` / ``use_full_rank_gate``.
    """

    cfg = dict(linear_attn_config or {})

    linear_hidden_size = (
        linear_hidden_size
        if linear_hidden_size is not None
        else int(cfg.get("linear_hidden_size", hidden_size))
    )
    linear_num_heads = (
        linear_num_heads
        if linear_num_heads is not None
        else int(cfg.get("linear_num_heads", cfg.get("num_heads", num_attention_heads)))
    )
    linear_head_dim = (
        linear_head_dim
        if linear_head_dim is not None
        else int(cfg.get("linear_head_dim", cfg.get("head_dim", 128)))
    )
    linear_conv_size = (
        linear_conv_size
        if linear_conv_size is not None
        else int(cfg.get("linear_conv_size", cfg.get("short_conv_kernel_size", 4)))
    )
    kda_use_full_rank_gate = (
        kda_use_full_rank_gate
        if kda_use_full_rank_gate is not None
        else bool(
            cfg.get(
                "kda_use_full_rank_gate",
                cfg.get("use_full_rank_gate", True),
            )
        )
    )

    if linear_hidden_size != hidden_size:
        raise ValueError(
            "Flash-KDA expects linear_hidden_size to match hidden_size; got "
            f"linear_hidden_size={linear_hidden_size}, hidden_size={hidden_size}."
        )
    if linear_num_heads <= 0 or linear_head_dim <= 0:
        raise ValueError("linear_num_heads and linear_head_dim must be positive")

    if cfg.get("kda_layers") is None:
        kda_layers, full_attn_layers = _resolve_hybrid_layer_pattern(
            num_hidden_layers,
            fa_interval=fa_interval,
            use_mla=use_mla,
        )
    else:
        kda_layers = list(cfg["kda_layers"])
        full_attn_layers = list(
            cfg.get(
                "full_attn_layers",
                [
                    i
                    for i in range(1, num_hidden_layers + 1)
                    if i not in set(kda_layers)
                ],
            )
        )

    normalized = dict(cfg)
    normalized.update(
        {
            "kda_layers": kda_layers,
            "full_attn_layers": full_attn_layers,
            "num_heads": linear_num_heads,
            "head_dim": linear_head_dim,
            "short_conv_kernel_size": linear_conv_size,
            "use_full_rank_gate": kda_use_full_rank_gate,
            "gate_lower_bound": cfg.get(
                "gate_lower_bound",
                cfg.get("linear_lower_bound", gate_lower_bound),
            ),
            # Flash / Megatron-facing aliases kept for config round-trip.
            "linear_hidden_size": linear_hidden_size,
            "linear_num_heads": linear_num_heads,
            "linear_head_dim": linear_head_dim,
            "linear_conv_size": linear_conv_size,
            "kda_use_full_rank_gate": kda_use_full_rank_gate,
        }
    )
    return normalized


def _default_linear_attn_config(num_hidden_layers: int) -> dict:
    """Default Flash-KDA KDA config: 3 KDA layers then 1 MLA layer."""

    return _normalize_linear_attn_config(
        None,
        num_hidden_layers=num_hidden_layers,
        hidden_size=3072,
        num_attention_heads=32,
        linear_hidden_size=3072,
        linear_head_dim=128,
        linear_num_heads=32,
        linear_conv_size=4,
        kda_use_full_rank_gate=True,
        fa_interval=4,
        use_mla=True,
    )


def _parse_special_token_scope(scope: str | None) -> list[int]:
    """Parse a ``special_token_scope`` string into a list of token ids.

    The scope is comma-separated ``"start:end"`` ranges (left-closed,
    right-open), e.g. ``"0:4,36:55"`` -> ``[0,1,2,3,36,...,54]``. These are the
    token ids excluded from OE n-gram hashing (mapped to the ignore row).
    """
    if scope is None or scope == "":
        return []
    ids: list[int] = []
    for id_range in scope.split(","):
        parts = id_range.split(":")
        id_start = int(parts[0])
        id_end = int(parts[1])
        ids.extend(range(id_start, id_end))
    return ids


class FLASHLocalConfig(PretrainedConfig):
    """Text-backbone configuration for Flash-KDA 3B (``model_type = "flash_kda"``).

    Carries five groups of fields:

    * **MLA** (``q_lora_rank`` .. ``mla_use_output_gate``) consumed by the
      full-attention layers and by ``configure_mla_attention``.
    * **MoE / EveryLayer-MoE** (``num_experts`` .. ``moe_group_size``) consumed
      by the Group-MoE block.
    * **KDA** (``linear_attn_config``) consumed by the linear-attention
      layers and, via the mixed-layer protocol properties below, by the hybrid
      KV-cache allocator.
    * **Group-MoE extras** (``moe_group_size`` .. ``zero_expert_type``)
      consumed by the grouped routing path.
    * **OE (Over-Embedding)** (``oe_vocab_size_ratio`` .. ``special_token_scope``)
      consumed by ``FusedOverEmbedding``.

    The ``layers_block_type`` / ``layer_types`` / ``linear_layer_ids`` /
    ``full_attention_layer_ids`` / ``mamba2_cache_params`` /
    ``mamba_cache_per_req`` properties are the interface consumed by the
    KV-cache / hybrid-attention layer (owned by the KV-cache team). Their
    shapes are derived from the KDA config here; the final state layout/dtype
    is validated on the cache side.
    """

    model_type = "flash_kda"

    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs):
        strict = _STRICT_FLASH_LITE_MARKERS.issubset(config_dict)
        if strict:
            missing = sorted(_STRICT_FLASH_LITE_FIELDS.difference(config_dict))
            if missing:
                raise ValueError(
                    "Flash-Lite checkpoint config is missing required fields: "
                    + ", ".join(missing)
                )
            if config_dict["architectures"] != ["FLASHLocalForCausalLM"]:
                raise ValueError(
                    "Flash-Lite architectures must be ['FLASHLocalForCausalLM']."
                )

        result = super().from_dict(config_dict, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        config.strict_checkpoint_layout = strict
        if strict:
            config._validate_strict_flash_lite()
        return result

    def __init__(
        self,
        vocab_size: int = 163840,
        hidden_size: int = 3072,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = 32,
        hidden_act: str = "silu",
        activation_situ_beta: float | None = 4.0,
        activation_situ_linear_beta: float | None = 25.0,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 8192,
        # Released Flash-Lite checkpoint fields, migrated from the removed
        # LiteConfig. Legacy Flash-KDA configs keep these no-op defaults.
        use_cache: bool = True,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        tie_word_embeddings: bool = False,
        torch_dtype="bfloat16",
        params_dtype="bfloat16",
        router_dtype="float32",
        router_bias=False,
        nextn_use_scmoe=False,
        # MLA
        q_lora_rank: int | None = 1536,
        kv_lora_rank: int | None = 512,
        qk_nope_head_dim: int | None = 128,
        qk_rope_head_dim: int | None = 64,
        v_head_dim: int | None = 128,
        mla_use_nope: bool | None = None,
        mla_use_output_gate: bool = True,
        mla_scale_q_lora: bool = False,
        mla_scale_kv_lora: bool = False,
        # Group MoE / EveryLayer-MoE
        n_routed_experts: int | None = 384,
        num_shared_experts: int | None = None,
        n_shared_experts: int | None = None,
        moe_intermediate_size: int | None = 512,
        moe_renormalize: bool = False,
        moe_router_activation_func: str = "softmax",
        topk_method: str = "noaux_tc",
        use_grouped_topk: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        routed_scaling_factor: float = 6.0,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        moe_topk: int | None = 12,
        moe_switch_token_num: int = 1024,
        moe_impl: str = "mix",
        # Group-MoE extras
        moe_group_size: int = 4,
        grouped_moe_norm_scale: int = 2,
        zero_expert_num: int = 32,
        zero_expert_type: str = "identity",
        # OE (Over-Embedding / n-gram embedding) — n-gram over-embedding
        # tables. Parameter names mirror the longcat-flash-lite open config:
        # ``ngram_vocab_size_ratio`` (OE table base size = ratio * vocab_size),
        # ``emb_split_num`` (= over_embedding_k, hash families per n-gram
        # order), ``emb_neighbor_num`` (= over_embedding_n, max n-gram order).
        # Legacy ``oe_*`` aliases are accepted for backward compatibility.
        ngram_vocab_size_ratio: float | None = None,
        emb_split_num: int | None = None,
        emb_neighbor_num: int | None = None,
        # ngram_exclude_sp_token: when True, special tokens (listed in
        # ``special_token_scope``) bypass OE and delimit the n-gram history.
        ngram_exclude_sp_token: bool = False,
        # special_token_scope: comma-separated "start:end" ranges (left-closed,
        # right-open) of token ids excluded from OE n-gram hashing,
        # e.g. "0:4,36:55". Parsed into ``oe_ignore_tokens`` at init time.
        special_token_scope: str | None = None,
        # ngram_fix_normalize_factor: the training implementation normalizes
        # word + OE by sqrt(1 + n_grams), rather than by (1 + n_grams).
        ngram_fix_normalize_factor: bool = False,
        # Legacy aliases (kept for backward compatibility with older configs).
        oe_vocab_size_ratio: float | None = None,
        oe_split_num: int | None = None,
        oe_neighbor_num: int | None = None,
        # KDA / linear attention
        linear_attn_config: dict | None = None,
        use_mla: int | bool = True,
        attention_method: str | None = None,
        fa_interval: int = 4,
        linear_hidden_size: int | None = None,
        linear_head_dim: int | None = None,
        linear_num_heads: int | None = None,
        linear_conv_size: int | None = None,
        linear_method: str = "FGBKDA",
        kda_use_full_rank_gate: bool | None = None,
        kda_nope: bool = True,
        linear_lower_bound: float = -5.0,
        rope_theta: float = 1_000_000.0,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_key_value_heads
            if num_key_value_heads is not None
            else num_attention_heads
        )
        self.hidden_act = hidden_act
        self.activation_situ_beta = activation_situ_beta
        self.activation_situ_linear_beta = activation_situ_linear_beta
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        if "num_layers" in kwargs and "num_hidden_layers" not in kwargs:
            self.num_hidden_layers = kwargs["num_layers"]
        if "ffn_hidden_size" in kwargs and "intermediate_size" not in kwargs:
            self.intermediate_size = kwargs["ffn_hidden_size"]
        if (
            "expert_ffn_hidden_size" in kwargs
            and moe_intermediate_size == 512
            and "moe_intermediate_size" not in kwargs
        ):
            moe_intermediate_size = kwargs["expert_ffn_hidden_size"]

        # MLA
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        # The released config uses ``kda_nope`` for the hybrid stack. The
        # training branch later renamed that switch to ``mla_nope``; both mean
        # that MLA layers skip RoPE. An explicit mla_use_nope still wins.
        self.mla_use_nope = bool(kda_nope) if mla_use_nope is None else mla_use_nope
        self.mla_use_output_gate = mla_use_output_gate
        self.mla_scale_q_lora = mla_scale_q_lora
        self.mla_scale_kv_lora = mla_scale_kv_lora
        self.use_mla = bool(use_mla)
        self.attention_method = attention_method or ("MLA" if self.use_mla else "KDA")
        self.fa_interval = fa_interval

        # Group MoE / EveryLayer-MoE
        self.n_routed_experts = n_routed_experts
        if n_shared_experts is None:
            n_shared_experts = 1 if num_shared_experts is None else num_shared_experts
        elif num_shared_experts is not None and num_shared_experts != n_shared_experts:
            raise ValueError(
                "Conflicting shared-expert counts: "
                f"num_shared_experts={num_shared_experts}, "
                f"n_shared_experts={n_shared_experts}"
            )
        self.n_shared_experts = n_shared_experts
        self.num_shared_experts = n_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_renormalize = moe_renormalize
        self.moe_router_activation_func = moe_router_activation_func
        if self.moe_router_activation_func != "softmax":
            raise ValueError(
                "Flash-KDA only supports softmax routing; got "
                f"{self.moe_router_activation_func!r}"
            )
        self.topk_method = topk_method
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.moe_topk = moe_topk
        self.moe_switch_token_num = moe_switch_token_num
        self.moe_impl = moe_impl

        # Group-MoE extras
        self.moe_group_size = moe_group_size
        self.grouped_moe_norm_scale = grouped_moe_norm_scale
        self.zero_expert_num = zero_expert_num
        self.zero_expert_type = zero_expert_type

        # OE (Over-Embedding / n-gram embedding).
        # Resolve legacy ``oe_*`` aliases to the canonical ``ngram_*`` / ``emb_*``
        # names so the rest of the runtime reads a single field set.
        ngram_vocab_size_ratio = (
            ngram_vocab_size_ratio
            if ngram_vocab_size_ratio is not None
            else oe_vocab_size_ratio
        )
        emb_split_num = emb_split_num if emb_split_num is not None else oe_split_num
        emb_neighbor_num = (
            emb_neighbor_num if emb_neighbor_num is not None else oe_neighbor_num
        )
        self.ngram_vocab_size_ratio = ngram_vocab_size_ratio
        self.emb_split_num = emb_split_num
        self.emb_neighbor_num = emb_neighbor_num
        self.ngram_exclude_sp_token = ngram_exclude_sp_token
        self.ngram_fix_normalize_factor = ngram_fix_normalize_factor
        self.use_over_embedding = ngram_vocab_size_ratio is not None
        self.over_embedding_m: int | None = None
        # oe_ignore_tokens: token ids excluded from OE n-gram hashing. Parsed
        # from ``special_token_scope`` (e.g. "0:4,36:55" -> [0,1,2,3,36,...,54]).
        # Only effective when ``ngram_exclude_sp_token`` is True.
        self.oe_ignore_tokens: list[int] = _parse_special_token_scope(
            special_token_scope
        )
        if self.use_over_embedding:
            self.over_embedding_m = int(vocab_size * ngram_vocab_size_ratio)
        self.oe_vocab_size_ratio = self.ngram_vocab_size_ratio
        self.oe_split_num = self.emb_split_num
        self.oe_neighbor_num = self.emb_neighbor_num
        self.special_token_scope = special_token_scope

        # KDA / linear attention
        self.linear_hidden_size = (
            linear_hidden_size if linear_hidden_size is not None else hidden_size
        )
        self.linear_head_dim = (
            linear_head_dim if linear_head_dim is not None else qk_nope_head_dim
        )
        self.linear_num_heads = (
            linear_num_heads if linear_num_heads is not None else num_attention_heads
        )
        self.linear_conv_size = linear_conv_size if linear_conv_size is not None else 4
        self.kda_use_full_rank_gate = (
            True if kda_use_full_rank_gate is None else kda_use_full_rank_gate
        )
        self.linear_attn_config = _normalize_linear_attn_config(
            linear_attn_config,
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            linear_hidden_size=self.linear_hidden_size,
            linear_head_dim=self.linear_head_dim,
            linear_num_heads=self.linear_num_heads,
            linear_conv_size=self.linear_conv_size,
            kda_use_full_rank_gate=self.kda_use_full_rank_gate,
            fa_interval=self.fa_interval,
            use_mla=self.use_mla,
            gate_lower_bound=linear_lower_bound,
        )
        # full-attention layers are derived as the complement of kda_layers
        # (see full_attention_layer_ids), so only kda_layers is required here.
        if self.linear_attn_config.get("kda_layers") is None:
            raise ValueError("linear_attn_config must provide 'kda_layers'")
        self.linear_method = linear_method
        self.kda_nope = kda_nope
        self.linear_lower_bound = self.linear_attn_config["gate_lower_bound"]

        # ``num_experts`` is the runtime-facing alias; prefer ``n_routed_experts``
        # from the checkpoint when both are present.
        self.num_experts = n_routed_experts
        self.ffn_hidden_size = self.intermediate_size
        self.norm_topk_prob = moe_renormalize
        self.num_experts_per_token = moe_topk
        self.expert_ffn_hidden_size = moe_intermediate_size

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            torch_dtype=torch_dtype,
            params_dtype=params_dtype,
            router_dtype=router_dtype,
            topk_method=topk_method,
            router_bias=router_bias,
            nextn_use_scmoe=nextn_use_scmoe,
            rope_theta=rope_theta,
            **kwargs,
        )

    def _validate_strict_flash_lite(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "ffn_hidden_size": self.ffn_hidden_size,
            "expert_ffn_hidden_size": self.expert_ffn_hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "linear_head_dim": self.linear_head_dim,
            "linear_num_heads": self.linear_num_heads,
            "linear_conv_size": self.linear_conv_size,
            "fa_interval": self.fa_interval,
            "moe_group_size": self.moe_group_size,
            "n_routed_experts": self.n_routed_experts,
            "zero_expert_num": self.zero_expert_num,
            "moe_topk": self.moe_topk,
            "moe_switch_token_num": self.moe_switch_token_num,
            "emb_neighbor_num": self.emb_neighbor_num,
            "emb_split_num": self.emb_split_num,
            "ngram_vocab_size_ratio": self.ngram_vocab_size_ratio,
            "grouped_moe_norm_scale": self.grouped_moe_norm_scale,
        }
        invalid = [
            name for name, value in positive.items() if value is None or value <= 0
        ]
        if invalid:
            raise ValueError(
                "Flash-Lite config fields must be positive: " + ", ".join(invalid)
            )
        if self.num_hidden_layers % self.fa_interval:
            raise ValueError("num_layers must be divisible by fa_interval.")
        if self.linear_hidden_size != self.hidden_size:
            raise ValueError("linear_hidden_size must equal hidden_size.")
        if self.linear_num_heads % 8:
            raise ValueError("linear_num_heads must be divisible by KDA TP8.")
        if (self.n_routed_experts * self.moe_group_size) % 8:
            raise ValueError("Grouped MoE real experts must be divisible by EP8.")
        if self.hidden_size % self.moe_group_size:
            raise ValueError("hidden_size must be divisible by moe_group_size.")
        if self.moe_topk > self.n_routed_experts + self.zero_expert_num:
            raise ValueError("moe_topk exceeds the experts available in one group.")
        if self.linear_method.upper() != "FGBKDA" or not self.kda_use_full_rank_gate:
            raise ValueError("Flash-Lite requires FGBKDA with the full-rank gate.")
        if not self.kda_nope:
            raise ValueError("Flash-Lite requires NoPE MLA.")
        if not self.use_mla or self.attention_method.upper() != "MLA":
            raise ValueError("Flash-Lite requires the hybrid MLA attention path.")
        if self.zero_expert_type != "identity":
            raise ValueError("Flash-Lite zero experts must use identity semantics.")
        if self.moe_impl != "mix" or not self.use_cache:
            raise ValueError("Flash-Lite requires mix MoE and cache support.")
        if self.attention_bias or self.attention_dropout != 0:
            raise ValueError(
                "Flash-Lite attention must be bias-free with zero dropout."
            )
        if not (
            self.mla_scale_q_lora
            and self.mla_scale_kv_lora
            and self.ngram_exclude_sp_token
            and self.ngram_fix_normalize_factor
        ):
            raise ValueError(
                "Flash-Lite checkpoint scaling and OE switches must be enabled."
            )
        self.special_token_ids

    # ----- helpers -----

    def is_kda_layer(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` (0-based) is a KDA linear-attention layer.

        ``linear_attn_config["kda_layers"]`` lists 1-based layer numbers, so a
        0-based index maps via ``layer_idx + 1``.
        """
        return (layer_idx + 1) in self.linear_attn_config["kda_layers"]

    # ----- mixed-layer protocol (consumed by the KV-cache / hybrid layer) -----

    @property
    def layers_block_type(self) -> list[str]:
        """Per-layer type: ``"linear_attention"`` (KDA) or ``"attention"`` (MLA)."""
        return [
            (_LINEAR_ATTENTION_LAYER if self.is_kda_layer(i) else _ATTENTION_LAYER)
            for i in range(self.num_hidden_layers)
        ]

    @property
    def layer_types(self) -> list[str]:
        """``layers_block_type`` translated to paged-cache labels."""
        return [
            FULL_ATTENTION if layer_type == _ATTENTION_LAYER else layer_type
            for layer_type in self.layers_block_type
        ]

    @property
    def linear_layer_ids(self) -> list[int]:
        return [
            i
            for i, t in enumerate(self.layers_block_type)
            if t == _LINEAR_ATTENTION_LAYER
        ]

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return [
            i for i, t in enumerate(self.layers_block_type) if t == _ATTENTION_LAYER
        ]

    @property
    def oe_component_count(self) -> int:
        if self.emb_neighbor_num is None or self.emb_split_num is None:
            raise ValueError("OE requires emb_neighbor_num and emb_split_num.")
        return (self.emb_neighbor_num - 1) * self.emb_split_num

    @property
    def oe_hidden_size(self) -> int:
        count = self.oe_component_count
        if self.hidden_size % count:
            raise ValueError("hidden_size must be divisible by the OE component count.")
        return self.hidden_size // count

    @property
    def oe_table_base_rows(self) -> int:
        if self.ngram_vocab_size_ratio is None:
            raise ValueError("OE requires ngram_vocab_size_ratio.")
        return int(self.vocab_size * self.ngram_vocab_size_ratio)

    def oe_table_rows(self, table_id: int) -> int:
        if not 0 <= table_id < self.oe_component_count:
            raise IndexError(f"OE table index {table_id} is outside the model.")
        return self.oe_table_base_rows + 2 * table_id + 1

    @property
    def special_token_ids(self) -> tuple[int, ...]:
        return tuple(self.oe_ignore_tokens)

    @property
    def mamba2_cache_params(self):
        """KDA per-request state spec consumed by the hybrid KV-cache allocator.

        Returns ``(conv_state_shape, temporal_state_shape, conv_dtype,
        ssm_dtype, mamba_layer_ids)``. KDA runs three short causal convolutions
        (q/k/v), each ``num_heads * head_dim`` wide, and keeps a per-head
        ``head_dim x head_dim`` recurrent (delta-rule) state in fp32.

        NOTE: this is the interface for the KV-cache team; the concrete state
        layout is validated on the cache side.
        """
        # Imported lazily to avoid config/env import cycles at module load.
        from tokenspeed.runtime.utils.env import global_server_args_dict

        mapping = global_server_args_dict["mapping"]
        # KDA state follows the KDA head shard. ``mapping.attn`` may instead
        # describe an independent MLA CP/DP execution domain.
        linear_attn_tp_size = mapping.linear_attn.tp_size

        la = self.linear_attn_config
        num_heads = la["linear_num_heads"]
        head_dim = la["linear_head_dim"]
        conv_kernel_size = la["linear_conv_size"]

        conv_dim = 3 * num_heads * head_dim
        conv_state_shape = (
            divide(conv_dim, linear_attn_tp_size),
            conv_kernel_size - 1,
        )
        temporal_state_shape = (
            divide(num_heads, linear_attn_tp_size),
            head_dim,
            head_dim,
        )
        conv_dtype = torch.bfloat16
        # KDA recurrent (delta-rule) state is fp32.
        ssm_dtype = torch.float32
        return (
            conv_state_shape,
            temporal_state_shape,
            conv_dtype,
            ssm_dtype,
            self.linear_layer_ids,
        )

    @property
    def mamba_cache_per_req(self) -> int:
        conv_state_shape, temporal_state_shape, conv_dtype, ssm_dtype, mamba_layers = (
            self.mamba2_cache_params
        )
        return (
            math.prod(conv_state_shape) * conv_dtype.itemsize
            + math.prod(temporal_state_shape) * ssm_dtype.itemsize
        ) * len(mamba_layers)
