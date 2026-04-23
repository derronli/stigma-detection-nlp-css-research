"""Hierarchical transformer fusion for multi-turn context embeddings."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .dcn import DCN
from .dcnv2 import DCNv2

PosEncodingMode = Literal["distance", "slot"]
PoolMode = Literal["target", "mean"]


class HierarchicalContextFusion(nn.Module):
    """
    Args:
        turn_embeddings: [batch, num_turns, hidden_dim]
    Returns:
        [batch, 2 * hidden_dim] — concat(h_t, h_ctx)
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_roles: int = 3,
        num_distance_bins: int = 3,
        num_turns: int = 3,
        role_ids: tuple[int, ...] | None = None,
        distance_ids: tuple[int, ...] | None = None,
        target_turn_index: int = -1,
        num_context_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pos_encoding: PosEncodingMode = "distance",
        pool_mode: PoolMode = "target",
        target_proj: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % nhead != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by nhead ({nhead})")
        self.hidden_dim = hidden_dim
        self.num_turns = num_turns
        self.pos_encoding = pos_encoding
        self.pool_mode = pool_mode
        if target_turn_index < 0:
            target_turn_index = num_turns + target_turn_index
        if target_turn_index < 0 or target_turn_index >= num_turns:
            raise ValueError(f"target_turn_index out of range: {target_turn_index}")
        self.target_turn_index = target_turn_index

        self.role_emb = nn.Embedding(num_roles, hidden_dim)
        self.pos_emb = nn.Embedding(num_distance_bins, hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_context_layers)
        self.in_norm = nn.LayerNorm(hidden_dim)
        self.h_t_norm = nn.LayerNorm(hidden_dim)
        self.h_ctx_norm = nn.LayerNorm(hidden_dim)
        self.out_norm = nn.LayerNorm(2 * hidden_dim)

        self.target_proj = nn.Linear(hidden_dim, hidden_dim) if target_proj else None

        if role_ids is None:
            role_ids = tuple(range(num_turns))
        if distance_ids is None:
            distance_ids = tuple(reversed(range(num_turns)))
        if len(role_ids) != num_turns or len(distance_ids) != num_turns:
            raise ValueError("role_ids and distance_ids must match num_turns length")
        self.register_buffer("role_ids", torch.tensor(role_ids, dtype=torch.long), persistent=False)
        self.register_buffer("distance_ids", torch.tensor(distance_ids, dtype=torch.long), persistent=False)
        self.register_buffer(
            "slot_pos_ids", torch.tensor(tuple(range(num_turns)), dtype=torch.long), persistent=False
        )

    def _pos_ids(self) -> torch.Tensor:
        if self.pos_encoding == "distance":
            return self.distance_ids
        if self.pos_encoding == "slot":
            return self.slot_pos_ids
        raise ValueError(f"Unknown pos_encoding: {self.pos_encoding}")

    def forward(self, turn_embeddings: torch.Tensor) -> torch.Tensor:
        """
        turn_embeddings: [B, T, H] with T=num_turns
        """
        if turn_embeddings.dim() != 3:
            raise ValueError(f"Expected [B,T,H], got {tuple(turn_embeddings.shape)}")
        b, t, h = turn_embeddings.shape
        if t != self.num_turns or h != self.hidden_dim:
            raise ValueError(f"Expected [*, {self.num_turns}, {self.hidden_dim}], got {tuple(turn_embeddings.shape)}")

        role_ids = self.role_ids.view(1, -1).expand(b, -1)
        pos_ids = self._pos_ids().view(1, -1).expand(b, -1)

        x = turn_embeddings + self.role_emb(role_ids) + self.pos_emb(pos_ids)
        x = self.in_norm(x)
        out = self.encoder(x)

        h_t = turn_embeddings[:, self.target_turn_index, :]
        if self.target_proj is not None:
            h_t = self.target_proj(h_t)

        if self.pool_mode == "target":
            h_ctx = out[:, self.target_turn_index, :]
        elif self.pool_mode == "mean":
            h_ctx = out.mean(dim=1)
        else:
            raise ValueError(f"Unknown pool_mode: {self.pool_mode}")

        h_t = self.h_t_norm(h_t)
        h_ctx = self.h_ctx_norm(h_ctx)
        fused = torch.cat([h_t, h_ctx], dim=-1)
        return self.out_norm(fused)


class HierarchicalDCN(nn.Module):
    """Fusion (turn tensor -> text vec) + DCN or DCNv2."""

    def __init__(
        self,
        fusion: HierarchicalContextFusion,
        head: DCN | DCNv2,
    ) -> None:
        super().__init__()
        self.fusion = fusion
        self.head = head

    def forward(
        self,
        turn_embeddings: torch.Tensor,
        demo: torch.Tensor,
        worker: torch.Tensor,
    ) -> torch.Tensor:
        text_vec = self.fusion(turn_embeddings)
        return self.head(text_vec, demo, worker)


def build_fusion_from_config(cfg: dict) -> HierarchicalContextFusion:
    """Rebuild fusion from trainer checkpoint ``config`` dict."""
    n_turns = int(cfg.get("hier_num_turns", 2))
    role_default = list(range(n_turns))
    dist_default = list(reversed(range(n_turns)))
    role_ids = tuple(int(x) for x in cfg.get("hier_role_ids", role_default))
    distance_ids = tuple(int(x) for x in cfg.get("hier_distance_ids", dist_default))
    return HierarchicalContextFusion(
        hidden_dim=int(cfg.get("hier_hidden_dim", cfg.get("bert_hidden", 768))),
        num_turns=n_turns,
        role_ids=role_ids,
        distance_ids=distance_ids,
        target_turn_index=int(cfg.get("hier_target_turn_index", -1)),
        num_context_layers=int(cfg.get("hier_num_context_layers", 2)),
        nhead=int(cfg.get("hier_nhead", 8)),
        dim_feedforward=int(cfg.get("hier_dim_feedforward", 2048)),
        dropout=float(cfg.get("hier_dropout", cfg.get("dropout", 0.1))),
        pos_encoding=str(cfg.get("hier_pos_encoding", "distance")),  # type: ignore[arg-type]
        pool_mode=str(cfg.get("hier_pool_mode", "target")),  # type: ignore[arg-type]
        target_proj=bool(cfg.get("hier_target_proj", False)),
    )


def build_hierarchical_model(
    cfg: dict,
    num_workers: int,
    head_cls: type,
) -> HierarchicalDCN:
    fusion = build_fusion_from_config(cfg)
    hidden = fusion.hidden_dim
    text_dim = 2 * hidden
    demo_dim = int(cfg.get("demo_dim", 4))
    annot_dim = int(cfg.get("annotator_emb_dim", 8))
    input_dim = text_dim + demo_dim + annot_dim
    head = head_cls(
        input_dim=input_dim,
        num_workers=num_workers,
        annotator_emb_dim=annot_dim,
        text_dim=text_dim,
        demo_dim=demo_dim,
        num_cross_layers=int(cfg.get("num_cross_layers", 3)),
        num_deep_layers=int(cfg.get("num_deep_layers", 3)),
        deep_layer_size=int(cfg.get("deep_layer_size", 256)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    return HierarchicalDCN(fusion, head)
