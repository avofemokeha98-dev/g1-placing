"""Load frozen low-level (\"cerebellum\") policies from RSL-RL checkpoints.

Expected checkpoint layout (Isaac Lab / RSL-RL PPO):
``torch.save({\"model_state_dict\": ..., ...})`` where ``model_state_dict`` contains:
- ``actor.*`` weights for an MLP mapping obs dim -> action dim
- ``std`` Learnable std per action dim (same convention as rsl_rl OnPolicyRunner)

The placing cerebellum uses ``obs (N,106) -> joint_delta (N,37)`` with ELU MLP
``[512,256,128]`` per ``agents/rsl_rl_ppo_cfg.py``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class _PlacingCerebellum(nn.Module):
    """Deterministic actor matching exported RSL-RL actor layout."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: tuple[int, int, int]):
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ELU(),
            nn.Linear(h1, h2),
            nn.ELU(),
            nn.Linear(h2, h3),
            nn.ELU(),
            nn.Linear(h3, act_dim),
        )
        self.register_buffer("action_std", torch.ones(act_dim, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Match training-time action scaling: tanh(mean) * exp(log_std), where log_std = log(std)
        mean = self.net(obs)
        return torch.tanh(mean) * self.action_std


def load_frozen_policy(checkpoint_path: str | Path, device: str | torch.device) -> nn.Module:
    """Load a frozen placing policy network from an RSL-RL ``model_XXXX.pt`` file."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"cerebellum checkpoint not found: {path}")

    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Unexpected checkpoint format (missing model_state_dict): {path}")

    sd = payload["model_state_dict"]
    if "actor.0.weight" not in sd or "std" not in sd:
        raise ValueError(f"Checkpoint missing actor/std tensors: {path}")

    obs_dim = int(sd["actor.0.weight"].shape[1])
    act_dim = int(sd["std"].shape[0])

    # Hidden dims are encoded in weight shapes; validate expected placing dims.
    if obs_dim != 106 or act_dim != 37:
        raise ValueError(f"Unexpected cerebellum dims: obs={obs_dim}, act={act_dim} (expected 106/37)")

    policy = _PlacingCerebellum(obs_dim, act_dim, hidden_dims=(512, 256, 128)).to(device)

    actor_sd = {k[len("actor.") :]: v for k, v in sd.items() if k.startswith("actor.")}
    missing, unexpected = policy.net.load_state_dict(actor_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Failed to load actor weights cleanly: missing={missing} unexpected={unexpected}")

    policy.action_std.copy_(sd["std"].to(device=device, dtype=policy.action_std.dtype))

    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)

    return policy
