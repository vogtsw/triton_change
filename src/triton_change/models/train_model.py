from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from triton_change.models.hybrid_model import base_config, fake_batch, make_model, modified_config


def train_tiny(variant: str, steps: int, out: Path) -> Path:
    config = modified_config() if variant == "modified" else base_config()
    model = make_model(config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(steps):
        input_ids, labels = fake_batch(config, batch_size=4)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(input_ids), labels)
        loss.backward()
        optimizer.step()

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"variant": variant, "config": config.__dict__, "state_dict": model.state_dict()}, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the tiny sample model on synthetic data.")
    parser.add_argument("--variant", choices=["base", "modified"], default="base")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("artifacts/checkpoints/model.pt"))
    args = parser.parse_args()
    path = train_tiny(args.variant, args.steps, args.out)
    print(path)


if __name__ == "__main__":
    main()

