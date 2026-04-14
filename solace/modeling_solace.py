"""Claude Solace transformer.

A decoder-only LM combining ideas from the Gemma family (1 → 2 → 3, with the
trajectory pointed at "Gemma 4") and the GPT-OSS family:

  * GeGLU MLPs (Gemma).
  * Pre + post RMSNorm sandwich on every sublayer (Gemma 2/3).
  * QK-norm before RoPE (Gemma 3).
  * Sliding-window attention alternated with full attention every 6th layer
    (Gemma 3 5:1 pattern).
  * Separate RoPE base for local (10 000) and global (1 000 000) layers
    (Gemma 3 dual-RoPE).
  * Tied embeddings, large vocab (~262k) (Gemma).
  * Optional logit softcapping (Gemma 2).
  * Attention sinks: a small number of permanently-attended register tokens
    that stabilise streaming inference (GPT-OSS).
  * Grouped-query attention with bf16 K/V cache (Claude/GPT-OSS/Gemma).

Dimensions are picked per ``configs/solace_*.json`` so the same code runs
the local 16 GB, mid 24 GB, and HF 32 GB variants.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel

from .configuration_solace import SolaceConfig


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------

class SolaceRMSNorm(nn.Module):
    """RMSNorm with Gemma's ``(1 + weight)`` scale parameterisation."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x * (1.0 + self.weight.float())
        return x.to(dtype)


# ---------------------------------------------------------------------------
# Rotary position embeddings
# ---------------------------------------------------------------------------

class SolaceRotaryEmbedding(nn.Module):
    """RoPE with a configurable theta. One instance per attention layer so
    local vs. global layers can use different bases (Gemma 3)."""

    def __init__(self, head_dim: int, max_position: int, theta: float):
        super().__init__()
        self.head_dim = head_dim
        self.max_position = max_position
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # positions: (..., seq_len)
        freqs = torch.einsum("...i,j->...ij", positions.float(), self.inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: (B, H, T, D);  cos, sin: (B, T, D/2) -> broadcast
    cos = cos.unsqueeze(1).repeat_interleave(2, dim=-1)
    sin = sin.unsqueeze(1).repeat_interleave(2, dim=-1)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


# ---------------------------------------------------------------------------
# MLP (GeGLU / SwiGLU)
# ---------------------------------------------------------------------------

class SolaceMLP(nn.Module):
    def __init__(self, config: SolaceConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        if config.mlp_type == "geglu":
            # Gemma uses tanh-approx GELU on the gate.
            self.act = lambda x: F.gelu(x, approximate="tanh")
        else:
            self.act = F.silu  # SwiGLU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Attention (GQA + sliding-window + QK-norm + sinks + optional softcap)
# ---------------------------------------------------------------------------

def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


class SolaceAttention(nn.Module):
    def __init__(self, config: SolaceConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.kind = config.layer_attention_kind(layer_idx)  # 'local' or 'global'

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        h, kv_dim = config.hidden_size, config.kv_dim
        self.q_proj = nn.Linear(h, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(h, kv_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(h, kv_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, h, bias=config.attention_bias)

        if config.use_qk_norm:
            self.q_norm = SolaceRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = SolaceRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = self.k_norm = None

        self.rotary = SolaceRotaryEmbedding(
            head_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            theta=config.layer_rope_theta(layer_idx),
        )

        self.sliding_window = config.sliding_window if self.kind == "local" else None
        self.attention_sinks = config.attention_sinks
        self.attn_softcap = config.attn_logit_softcapping
        self.dropout = config.attention_dropout

    def _build_mask(
        self,
        q_len: int,
        kv_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Causal mask over the (q_len, kv_len) block, with sliding window
        # for local layers. The first ``attention_sinks`` keys are always
        # attendable (GPT-OSS sinks), so we exclude them from the window cut.
        i = torch.arange(q_len, device=device).unsqueeze(1) + (kv_len - q_len)
        j = torch.arange(kv_len, device=device).unsqueeze(0)
        causal = j <= i
        if self.sliding_window is not None:
            within_window = (i - j) < self.sliding_window
            sink = j < self.attention_sinks
            causal = causal & (within_window | sink)
        mask = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
        mask.masked_fill_(~causal, torch.finfo(dtype).min)
        return mask

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        B, T, _ = hidden.shape
        q = self.q_proj(hidden).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos, sin = self.rotary(positions)
        q, k = apply_rotary(q, k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

            # Sliding-window layers can prune the cache to (sinks + window).
            if self.sliding_window is not None:
                keep = self.sliding_window + self.attention_sinks
                if k.size(2) > keep:
                    sinks_k, sinks_v = k[:, :, : self.attention_sinks], v[:, :, : self.attention_sinks]
                    k = torch.cat([sinks_k, k[:, :, -self.sliding_window :]], dim=2)
                    v = torch.cat([sinks_v, v[:, :, -self.sliding_window :]], dim=2)

        present = (k, v) if use_cache else None

        # GQA: replicate K/V across query head groups
        k_full = _repeat_kv(k, self.kv_groups)
        v_full = _repeat_kv(v, self.kv_groups)

        attn_scores = torch.matmul(q, k_full.transpose(-1, -2)) * self.scale
        if self.attn_softcap is not None:
            attn_scores = self.attn_softcap * torch.tanh(attn_scores / self.attn_softcap)

        attn_scores = attn_scores + self._build_mask(
            q_len=q.size(2), kv_len=k_full.size(2),
            device=hidden.device, dtype=attn_scores.dtype,
        )
        attn = F.softmax(attn_scores, dim=-1)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        out = torch.matmul(attn, v_full).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out), present


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class SolaceDecoderLayer(nn.Module):
    def __init__(self, config: SolaceConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = SolaceRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = SolaceAttention(config, layer_idx)
        self.post_attention_layernorm = (
            SolaceRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_pre_post_norm_sandwich else nn.Identity()
        )

        self.pre_feedforward_layernorm = SolaceRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SolaceMLP(config)
        self.post_feedforward_layernorm = (
            SolaceRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_pre_post_norm_sandwich else nn.Identity()
        )

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        past_kv=None,
        use_cache: bool = False,
    ):
        residual = hidden
        x = self.input_layernorm(hidden)
        x, present = self.self_attn(x, positions, past_kv=past_kv, use_cache=use_cache)
        x = self.post_attention_layernorm(x)
        hidden = residual + x

        residual = hidden
        x = self.pre_feedforward_layernorm(hidden)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        hidden = residual + x
        return hidden, present


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class SolacePreTrainedModel(PreTrainedModel):
    config_class = SolaceConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SolaceDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=std)


class SolaceModel(SolacePreTrainedModel):
    def __init__(self, config: SolaceConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(
            [SolaceDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = SolaceRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_scale = math.sqrt(config.hidden_size)  # Gemma-style embedding scale
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,  # accepted for API parity
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: Optional[bool] = None,
        return_dict: bool = True,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        B, T = input_ids.shape

        past_len = past_key_values[0][0].size(2) if past_key_values is not None else 0
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + T, device=input_ids.device).unsqueeze(0).expand(B, -1)

        hidden = self.embed_tokens(input_ids) * self.embed_scale

        new_cache = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden, present = layer(hidden, position_ids, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                new_cache.append(present)

        hidden = self.norm(hidden)
        if not return_dict:
            return (hidden, new_cache)
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=new_cache)


class SolaceForCausalLM(SolacePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: SolaceConfig):
        super().__init__(config)
        self.model = SolaceModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: bool = True,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )
        logits = self.lm_head(outputs.last_hidden_state)

        cap = self.config.final_logit_softcapping
        if cap is not None:
            logits = cap * torch.tanh(logits / cap)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        if not return_dict:
            return (loss, logits, outputs.past_key_values) if loss is not None else (logits, outputs.past_key_values)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )
