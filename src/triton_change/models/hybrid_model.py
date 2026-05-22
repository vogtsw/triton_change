from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class HybridConfig:
    vocab_size: int = 257
    seq_len: int = 32
    embed_dim: int = 32
    conv_channels: int = 48
    transformer_heads: int = 4
    transformer_ff: int = 64
    transformer_layers: int = 1
    num_classes: int = 5
    insert_precision_cast: bool = False


def base_config() -> HybridConfig:
    return HybridConfig()


def modified_config() -> HybridConfig:
    return HybridConfig(
        seq_len=40,
        conv_channels=64,
        transformer_ff=96,
        insert_precision_cast=True,
    )


class HybridTextConvTransformer(nn.Module):
    """Tiny CPU-friendly model that exports Conv1d and Transformer patterns to ONNX."""

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.conv = nn.Conv1d(
            in_channels=config.embed_dim,
            out_channels=config.conv_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv_projection = nn.Linear(config.conv_channels, config.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.classifier = nn.Linear(config.embed_dim, config.num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        if self.config.insert_precision_cast:
            # Demonstrates a precision-local graph change without requiring GPU kernels.
            embedded = embedded.to(torch.float16).to(torch.float32)

        conv_features = self.conv(embedded.transpose(1, 2)).transpose(1, 2)
        conv_features = torch.relu(self.conv_projection(conv_features))
        transformer_features = self.transformer(embedded)
        pooled = self.norm(conv_features + transformer_features).mean(dim=1)
        return self.classifier(pooled)


def make_model(config: HybridConfig, seed: int = 7) -> HybridTextConvTransformer:
    torch.manual_seed(seed)
    model = HybridTextConvTransformer(config)
    model.eval()
    return model


def fake_batch(config: HybridConfig, batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, config.seq_len), dtype=torch.long)
    labels = torch.randint(0, config.num_classes, (batch_size,), dtype=torch.long)
    return input_ids, labels

