import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Position-wise FFN with configurable activation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.activation = activation.lower()
        self.dropout = nn.Dropout(dropout)

        if self.activation == "swiglu":
            self.fc1 = nn.Linear(d_model, d_ff * 2)
            self.fc2 = nn.Linear(d_ff, d_model)
        elif self.activation in {"relu", "gelu"}:
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
        else:
            raise ValueError("activation must be one of: relu, gelu, swiglu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            x = F.relu(self.fc1(x))
        elif self.activation == "gelu":
            x = F.gelu(self.fc1(x))
        else:
            # SwiGLU splits the hidden projection into value and gate parts.
            value, gate = self.fc1(x).chunk(2, dim=-1)
            x = F.silu(gate) * value

        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)

