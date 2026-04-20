from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..interfaces import DecisionModel
from ..types import GenerationState, SurrogateDecision


class UniformDecisionModel(DecisionModel):
    """Deterministic placeholder that always chooses the first surrogate."""

    def score(
        self,
        state: GenerationState,
        surrogate_names: list[str],
    ) -> SurrogateDecision:
        del state

        if not surrogate_names:
            raise ValueError("surrogate_names must not be empty.")

        goodness = {name: 0.0 for name in surrogate_names}
        chosen = surrogate_names[0]
        return SurrogateDecision(
            goodness=goodness,
            chosen_surrogate_name=chosen,
            metadata={"model_type": "uniform_decision_model"},
        )


@dataclass(slots=True)
class PFNBackboneConfig:
    hidden_dim: int = 256
    num_heads: int = 8
    num_context_layers: int = 4
    num_candidate_layers: int = 2
    num_action_layers: int = 2
    ff_multiplier: int = 4
    dropout: float = 0.1
    activation: str = "gelu"
    use_type_embeddings: bool = True
    max_action_tokens: int = 64


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        act = _make_activation(activation)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerFFN(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        ff_multiplier: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        ff_dim = ff_multiplier * hidden_dim
        act = _make_activation(activation)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    """Standard pre-norm transformer block."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_heads: int,
        ff_multiplier: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = TransformerFFN(
            hidden_dim=hidden_dim,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
            activation=activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(
            query=h,
            key=h,
            value=h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block: query attends to memory."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        num_heads: int,
        ff_multiplier: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.norm_q1 = nn.LayerNorm(hidden_dim)
        self.norm_kv1 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm_q2 = nn.LayerNorm(hidden_dim)
        self.ffn = TransformerFFN(
            hidden_dim=hidden_dim,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
            activation=activation,
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        memory_tokens: torch.Tensor,
        *,
        memory_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        q = self.norm_q1(query_tokens)
        kv = self.norm_kv1(memory_tokens)
        attn_out, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        query_tokens = query_tokens + self.dropout1(attn_out)
        query_tokens = query_tokens + self.ffn(self.norm_q2(query_tokens))
        return query_tokens


class SetConditionedPFNBackbone(nn.Module):
    """
    PFN-style set-to-action backbone.

    Inputs
    ------
    context_x : [B, N, Dc]
        Context token features from previous true evaluations.
    context_y : [B, N, 1]
        Scalar targets for context tokens.
    candidate_x : [B, Q, Dk]
        Current candidate population tokens.
    action_ids : [B, A]
        Integer ids of action / bundle tokens.
    context_mask : [B, N] or None
        1 for valid context tokens, 0 for padded tokens.
    candidate_mask : [B, Q] or None
        1 for valid candidate tokens, 0 for padded tokens.

    Outputs
    -------
    logits : [B, A]
        One logit per action token.
    """

    def __init__(
        self,
        context_dim: int,
        candidate_dim: int,
        config: PFNBackboneConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or PFNBackboneConfig()
        self.context_dim = int(context_dim)
        self.candidate_dim = int(candidate_dim)

        if self.context_dim < 0:
            raise ValueError(f"context_dim must be >= 0, got {self.context_dim}")
        if self.candidate_dim <= 0:
            raise ValueError(f"candidate_dim must be > 0, got {self.candidate_dim}")
        if self.config.max_action_tokens <= 0:
            raise ValueError("max_action_tokens must be > 0")

        h = self.config.hidden_dim
        act = self.config.activation
        dropout = self.config.dropout

        self.context_encoder = MLPBlock(
            in_dim=self.context_dim + 1,
            out_dim=h,
            hidden_dim=h,
            dropout=dropout,
            activation=act,
        )
        self.candidate_encoder = MLPBlock(
            in_dim=self.candidate_dim,
            out_dim=h,
            hidden_dim=h,
            dropout=dropout,
            activation=act,
        )

        self.action_embedding = nn.Embedding(self.config.max_action_tokens, h)

        if self.config.use_type_embeddings:
            self.context_type_embedding = nn.Parameter(torch.zeros(1, 1, h))
            self.candidate_type_embedding = nn.Parameter(torch.zeros(1, 1, h))
            self.action_type_embedding = nn.Parameter(torch.zeros(1, 1, h))
        else:
            self.register_parameter("context_type_embedding", None)
            self.register_parameter("candidate_type_embedding", None)
            self.register_parameter("action_type_embedding", None)

        self.context_layers = nn.ModuleList(
            [
                SelfAttentionBlock(
                    h,
                    num_heads=self.config.num_heads,
                    ff_multiplier=self.config.ff_multiplier,
                    dropout=dropout,
                    activation=act,
                )
                for _ in range(self.config.num_context_layers)
            ]
        )
        self.candidate_layers = nn.ModuleList(
            [
                SelfAttentionBlock(
                    h,
                    num_heads=self.config.num_heads,
                    ff_multiplier=self.config.ff_multiplier,
                    dropout=dropout,
                    activation=act,
                )
                for _ in range(self.config.num_candidate_layers)
            ]
        )

        self.action_context_layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    h,
                    num_heads=self.config.num_heads,
                    ff_multiplier=self.config.ff_multiplier,
                    dropout=dropout,
                    activation=act,
                )
                for _ in range(self.config.num_action_layers)
            ]
        )
        self.action_candidate_layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    h,
                    num_heads=self.config.num_heads,
                    ff_multiplier=self.config.ff_multiplier,
                    dropout=dropout,
                    activation=act,
                )
                for _ in range(self.config.num_action_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(h)
        self.head = nn.Sequential(
            nn.Linear(h, h),
            _make_activation(act),
            nn.Dropout(dropout),
            nn.Linear(h, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.02)

        if self.context_type_embedding is not None:
            nn.init.normal_(self.context_type_embedding, mean=0.0, std=0.02)
        if self.candidate_type_embedding is not None:
            nn.init.normal_(self.candidate_type_embedding, mean=0.0, std=0.02)
        if self.action_type_embedding is not None:
            nn.init.normal_(self.action_type_embedding, mean=0.0, std=0.02)

    @staticmethod
    def _make_key_padding_mask(
        mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        name: str,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.shape != (batch_size, seq_len):
            raise ValueError(f"{name} must have shape {(batch_size, seq_len)}, got {tuple(mask.shape)}")
        return mask <= 0.0

    def forward(
        self,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        candidate_x: torch.Tensor,
        action_ids: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_x.ndim != 3:
            raise ValueError(f"context_x must have shape [B, N, Dc], got {tuple(context_x.shape)}")
        if context_y.ndim != 3:
            raise ValueError(f"context_y must have shape [B, N, 1], got {tuple(context_y.shape)}")
        if candidate_x.ndim != 3:
            raise ValueError(f"candidate_x must have shape [B, Q, Dk], got {tuple(candidate_x.shape)}")
        if action_ids.ndim != 2:
            raise ValueError(f"action_ids must have shape [B, A], got {tuple(action_ids.shape)}")

        batch_size, context_len, context_dim = context_x.shape
        b2, candidate_len, candidate_dim = candidate_x.shape
        b3, num_actions = action_ids.shape

        if b2 != batch_size or b3 != batch_size:
            raise ValueError("All batch dimensions must match.")
        if context_dim != self.context_dim:
            raise ValueError(f"context_x last dimension must be {self.context_dim}, got {context_dim}")
        if candidate_dim != self.candidate_dim:
            raise ValueError(f"candidate_x last dimension must be {self.candidate_dim}, got {candidate_dim}")
        if context_y.shape != (batch_size, context_len, 1):
            raise ValueError(f"context_y must have shape {(batch_size, context_len, 1)}, got {tuple(context_y.shape)}")
        if num_actions == 0:
            raise ValueError("action_ids must contain at least one action token.")
        if torch.any(action_ids < 0) or torch.any(action_ids >= self.config.max_action_tokens):
            raise ValueError("action_ids contain out-of-range indices.")

        context_input = torch.cat([context_x, context_y], dim=-1)
        context_tokens = self.context_encoder(context_input)
        candidate_tokens = self.candidate_encoder(candidate_x)
        action_tokens = self.action_embedding(action_ids)

        if self.context_type_embedding is not None:
            context_tokens = context_tokens + self.context_type_embedding
            candidate_tokens = candidate_tokens + self.candidate_type_embedding
            action_tokens = action_tokens + self.action_type_embedding

        context_key_padding_mask = self._make_key_padding_mask(context_mask, batch_size, context_len, "context_mask")
        candidate_key_padding_mask = self._make_key_padding_mask(
            candidate_mask, batch_size, candidate_len, "candidate_mask"
        )

        if context_len > 0:
            for layer in self.context_layers:
                context_tokens = layer(
                    context_tokens,
                    key_padding_mask=context_key_padding_mask,
                )

        if candidate_len > 0:
            for layer in self.candidate_layers:
                candidate_tokens = layer(
                    candidate_tokens,
                    key_padding_mask=candidate_key_padding_mask,
                )

        for layer in self.action_context_layers:
            if context_len > 0:
                action_tokens = layer(
                    action_tokens,
                    context_tokens,
                    memory_key_padding_mask=context_key_padding_mask,
                )
            else:
                action_tokens = action_tokens + layer.ffn(layer.norm_q2(action_tokens))

        for layer in self.action_candidate_layers:
            if candidate_len > 0:
                action_tokens = layer(
                    action_tokens,
                    candidate_tokens,
                    memory_key_padding_mask=candidate_key_padding_mask,
                )
            else:
                action_tokens = action_tokens + layer.ffn(layer.norm_q2(action_tokens))

        action_tokens = self.final_norm(action_tokens)
        logits = self.head(action_tokens).squeeze(-1)

        if logits.shape != (batch_size, num_actions):
            raise RuntimeError(f"Expected output shape {(batch_size, num_actions)}, got {tuple(logits.shape)}")
        return logits
