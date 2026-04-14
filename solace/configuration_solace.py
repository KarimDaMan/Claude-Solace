"""Config class for Claude Solace.

The architecture is a hybrid of:
  * Gemma 2/3 (and the public Gemma-family trajectory): GeGLU MLP, pre+post
    RMSNorm sandwich on every sublayer, QK-norm before RoPE, sliding-window
    attention alternated with full attention, separate RoPE base for local
    vs. global layers, tied embeddings, large vocab.
  * GPT-OSS family: attention sinks (a small number of permanently-attended
    register tokens) for stable streaming, SwiGLU/GeGLU dense MLPs,
    grouped-query attention.
  * Claude: long-context-first design, helpful-by-default chat formatting,
    BF16 native weights.

The same code backs three sized variants:

    variant   target RAM   params      quant     native ctx
    --------  ----------   ---------   -------   ----------
    local       16 GB      ~3.83B      Q4_K_M       32 768
    mid         24 GB      ~6.73B      BF16/INT8    65 536
    hf          32 GB      ~12.34B     BF16        131 072

The ``configs/solace_*.json`` files load as standard HF configs.
"""

from __future__ import annotations

from typing import Iterable

from transformers.configuration_utils import PretrainedConfig


class SolaceConfig(PretrainedConfig):
    model_type = "solace"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # ------- core dimensions -------
        vocab_size: int = 262144,
        hidden_size: int = 3072,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 30,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 8,
        head_dim: int | None = None,
        # ------- mlp / norm -------
        hidden_act: str = "gelu_pytorch_tanh",
        mlp_type: str = "geglu",            # "geglu" (Gemma) or "swiglu"
        rms_norm_eps: float = 1e-6,
        use_qk_norm: bool = True,
        use_pre_post_norm_sandwich: bool = True,
        # ------- attention pattern -------
        attention_pattern: Iterable[str] | None = None,
        sliding_window: int = 4096,
        rope_theta_local: float = 10_000.0,
        rope_theta_global: float = 1_000_000.0,
        rope_scaling: dict | None = None,
        # ------- streaming / capping -------
        attention_sinks: int = 4,
        final_logit_softcapping: float | None = None,
        attn_logit_softcapping: float | None = None,
        # ------- context / training -------
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        attention_dropout: float = 0.0,
        # ------- tokens -------
        bos_token_id: int = 2,
        eos_token_id: int = 1,
        pad_token_id: int = 0,
        **kwargs,
    ):
        # Default Gemma-3-style 5 local : 1 global pattern.
        if attention_pattern is None:
            attention_pattern = ["local"] * 5 + ["global"]
        attention_pattern = list(attention_pattern)
        for p in attention_pattern:
            if p not in ("local", "global"):
                raise ValueError(
                    f"attention_pattern entries must be 'local' or 'global', got {p!r}"
                )

        if mlp_type not in ("geglu", "swiglu"):
            raise ValueError(f"mlp_type must be 'geglu' or 'swiglu', got {mlp_type!r}")

        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be a multiple "
                f"of num_key_value_heads ({num_key_value_heads}) for GQA."
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads

        self.hidden_act = hidden_act
        self.mlp_type = mlp_type
        self.rms_norm_eps = rms_norm_eps
        self.use_qk_norm = use_qk_norm
        self.use_pre_post_norm_sandwich = use_pre_post_norm_sandwich

        self.attention_pattern = attention_pattern
        self.sliding_window = sliding_window
        self.rope_theta_local = rope_theta_local
        self.rope_theta_global = rope_theta_global
        self.rope_scaling = rope_scaling

        self.attention_sinks = attention_sinks
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping

        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Helpers used by the modeling code and the memory estimator
    # ------------------------------------------------------------------

    def layer_attention_kind(self, layer_idx: int) -> str:
        """Return 'local' or 'global' for a given layer index."""
        return self.attention_pattern[layer_idx % len(self.attention_pattern)]

    def layer_rope_theta(self, layer_idx: int) -> float:
        return (
            self.rope_theta_global
            if self.layer_attention_kind(layer_idx) == "global"
            else self.rope_theta_local
        )

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def approx_parameters(self) -> int:
        """Closed-form parameter count for this config (excludes biases)."""
        v, d = self.vocab_size, self.hidden_size
        n, f = self.num_hidden_layers, self.intermediate_size
        kv_dim = self.kv_dim

        embed = v * d
        lm_head = 0 if self.tie_word_embeddings else v * d

        # Attention: q, o (full d x d) + k, v (d x kv_dim)
        attn = 2 * d * d + 2 * d * kv_dim
        # GeGLU / SwiGLU MLP: gate, up, down
        mlp = 3 * d * f
        # Norms: pre+post on attention and MLP if sandwich, else just pre
        per_block_norms = (4 if self.use_pre_post_norm_sandwich else 2) * d
        # QK-norm (per-head)
        qk_norm = (2 * self.head_dim) if self.use_qk_norm else 0

        per_layer = attn + mlp + per_block_norms + qk_norm
        final_norm = d
        return embed + lm_head + n * per_layer + final_norm

    def estimate_runtime_bytes(
        self,
        context_length: int | None = None,
        weight_dtype_bytes: float = 2.0,
        kv_dtype_bytes: float = 2.0,
        activation_overhead_bytes: int = 1_500_000_000,
    ) -> dict:
        """Estimate inference-time memory footprint.

        Sliding-window layers cap their KV at ``sliding_window`` tokens;
        only ``global`` layers grow with context length.
        """
        ctx = context_length or self.max_position_embeddings

        weight_bytes = int(self.approx_parameters * weight_dtype_bytes)

        per_token_kv = 2 * self.kv_dim * kv_dtype_bytes  # K + V

        n_global = sum(
            1 for i in range(self.num_hidden_layers)
            if self.layer_attention_kind(i) == "global"
        )
        n_local = self.num_hidden_layers - n_global
        local_tokens = min(ctx, self.sliding_window) + self.attention_sinks

        kv_bytes = int(
            n_global * ctx * per_token_kv
            + n_local * local_tokens * per_token_kv
        )

        return {
            "weights_bytes": weight_bytes,
            "kv_cache_bytes": kv_bytes,
            "activation_bytes": activation_overhead_bytes,
            "total_bytes": weight_bytes + kv_bytes + activation_overhead_bytes,
            "context_length": ctx,
            "global_layers": n_global,
            "local_layers": n_local,
        }
