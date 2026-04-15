import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertMLP(nn.Module):
    """A simple expert network used inside MoE FFN."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.activation = activation.lower()
        self.dropout = nn.Dropout(dropout)

        if self.activation == "swiglu":
            self.fc1 = nn.Linear(d_model, hidden_dim * 2)
            self.fc2 = nn.Linear(hidden_dim, d_model)
        elif self.activation in {"relu", "gelu"}:
            self.fc1 = nn.Linear(d_model, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, d_model)
        else:
            raise ValueError("activation must be one of: relu, gelu, swiglu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            x = F.relu(self.fc1(x))
        elif self.activation == "gelu":
            x = F.gelu(self.fc1(x))
        else:
            value, gate = self.fc1(x).chunk(2, dim=-1)
            x = F.silu(gate) * value

        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class TopKRouter(nn.Module):
    """Route each token to top-k experts."""

    def __init__(self, d_model: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k must be <= num_experts.")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.router(x)
        router_probs = torch.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return router_probs, topk_indices, topk_weights


class MoEFeedForward(nn.Module):
    """
    Token-level Mixture of Experts FFN.

    Each token is routed to top-k experts, and the expert outputs are fused
    using the normalized router weights.
    """

    def __init__(
        self,
        d_model: int,
        expert_hidden_dim: int,
        dropout: float,
        activation: str,
        num_experts: int,
        top_k_experts: int,
        use_aux_loss: bool = True,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.use_aux_loss = use_aux_loss
        self.router = TopKRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k_experts,
        )
        self.experts = nn.ModuleList(
            [
                ExpertMLP(
                    d_model=d_model,
                    hidden_dim=expert_hidden_dim,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_experts)
            ]
        )

    def _load_balancing_loss(
        self,
        router_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        importance = router_probs.mean(dim=(0, 1))
        dispatch_mask = F.one_hot(topk_indices, num_classes=self.num_experts).float()
        load = dispatch_mask.sum(dim=(0, 1, 2))
        load = load / load.sum().clamp_min(1.0)

        # Lower is better. Uniform routing gives a smaller value than collapsing
        # onto only a few experts.
        return self.num_experts * torch.sum(importance * load)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.size()
        router_probs, topk_indices, topk_weights = self.router(x)

        flat_x = x.reshape(-1, d_model)
        flat_topk_indices = topk_indices.reshape(-1, self.top_k_experts)
        flat_topk_weights = topk_weights.reshape(-1, self.top_k_experts)
        flat_output = flat_x.new_zeros(flat_x.size(0), d_model)

        for expert_id, expert in enumerate(self.experts):
            assignment_mask = flat_topk_indices.eq(expert_id)
            if not assignment_mask.any():
                continue

            token_positions, topk_positions = assignment_mask.nonzero(as_tuple=True)
            expert_input = flat_x.index_select(0, token_positions)
            expert_output = expert(expert_input)
            expert_weight = flat_topk_weights[token_positions, topk_positions].unsqueeze(-1)

            flat_output.index_add_(0, token_positions, expert_output * expert_weight)

        output = flat_output.view(batch_size, seq_len, d_model)

        aux_loss = output.new_zeros(())
        if self.use_aux_loss:
            aux_loss = self._load_balancing_loss(router_probs, topk_indices)
        return output, aux_loss
