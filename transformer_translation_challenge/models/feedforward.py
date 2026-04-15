import torch
import torch.nn as nn
import torch.nn.functional as F

from models.moe import MoEFeedForward


class DenseFeedForward(nn.Module):
    """Standard dense position-wise FFN."""

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.activation == "relu":
            x = F.relu(self.fc1(x))
        elif self.activation == "gelu":
            x = F.gelu(self.fc1(x))
        else:
            value, gate = self.fc1(x).chunk(2, dim=-1)
            x = F.silu(gate) * value

        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x, x.new_zeros(())


def build_feed_forward(config) -> nn.Module:
    if config.use_moe:
        return MoEFeedForward(
            d_model=config.d_model,
            expert_hidden_dim=config.expert_hidden_dim,
            dropout=config.dropout,
            activation=config.ffn_activation,
            num_experts=config.num_experts,
            top_k_experts=config.top_k_experts,
            use_aux_loss=config.use_moe_aux_loss,
        )
    return DenseFeedForward(
        d_model=config.d_model,
        d_ff=config.d_ff,
        dropout=config.dropout,
        activation=config.ffn_activation,
    )


FeedForward = DenseFeedForward
