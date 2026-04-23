"""Deep & Cross Network classifier used in this project."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class CrossLayer(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        xw = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x0 * xw + self.bias + x


class DCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_workers: int,
        annotator_emb_dim: int,
        text_dim: int,
        demo_dim: int,
        num_cross_layers: int = 3,
        num_deep_layers: int = 3,
        deep_layer_size: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.demo_dim = demo_dim
        self.annotator_emb_dim = annotator_emb_dim
        self.input_dim = input_dim

        if input_dim != text_dim + demo_dim + annotator_emb_dim:
            raise ValueError(
                f"input_dim ({input_dim}) must equal text_dim + demo_dim + annotator_emb_dim "
                f"({text_dim + demo_dim + annotator_emb_dim})"
            )

        self.worker_emb = nn.Embedding(num_workers, annotator_emb_dim)
        self.cross_layers = nn.ModuleList(
            CrossLayer(input_dim) for _ in range(num_cross_layers)
        )

        deep_modules: List[nn.Module] = []
        in_d = input_dim
        for _ in range(num_deep_layers):
            deep_modules.append(nn.Linear(in_d, deep_layer_size))
            deep_modules.append(nn.ReLU())
            deep_modules.append(nn.Dropout(dropout))
            in_d = deep_layer_size
        self.deep = nn.Sequential(*deep_modules)

        self.final_linear = nn.Linear(input_dim + deep_layer_size, 1)

    def forward(
        self,
        text_emb: torch.Tensor,
        demo: torch.Tensor,
        worker: torch.Tensor,
    ) -> torch.Tensor:
        w = self.worker_emb(worker)
        x0 = torch.cat([text_emb, demo, w], dim=-1)

        x = x0
        for layer in self.cross_layers:
            x = layer(x0, x)
        x_cross = x
        x_deep = self.deep(x0)
        h = torch.cat([x_cross, x_deep], dim=-1)
        return self.final_linear(h).squeeze(-1)