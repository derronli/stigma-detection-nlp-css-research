import torch
import torch.nn as nn


class DCNv2CrossLayer(nn.Module):
    """DCNv2 cross layer with vector output: x_{l+1} = x_l + x0 * (W x_l + b)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, input_dim) * (1.0 / (input_dim**0.5)))
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        # [batch, d], [batch, d]
        out = x0 * (torch.matmul(xl, self.weight)) + self.bias + xl
        return out


class DCNv2(nn.Module):
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

        self.annotator_emb = nn.Embedding(num_workers, annotator_emb_dim)
        self.cross_layers = nn.ModuleList([DCNv2CrossLayer(input_dim) for _ in range(num_cross_layers)])

        deep_layers = []
        dim = input_dim
        for _ in range(num_deep_layers):
            deep_layers.append(nn.Linear(dim, deep_layer_size))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout))
            dim = deep_layer_size
        self.deep = nn.Sequential(*deep_layers)

        self.output = nn.Linear(deep_layer_size + input_dim, 1)

    def forward(
        self, text_embedding: torch.Tensor, demo_features: torch.Tensor, worker_ids: torch.Tensor
    ) -> torch.Tensor:
        annotator_embed = self.annotator_emb(worker_ids)
        x = torch.cat([text_embedding, demo_features, annotator_embed], dim=1)
        x0 = x

        x_cross = x
        for layer in self.cross_layers:
            x_cross = layer(x0, x_cross)
        x_deep = self.deep(x)

        x_cat = torch.cat([x_cross, x_deep], dim=1)
        logits = self.output(x_cat).squeeze(-1)
        return logits