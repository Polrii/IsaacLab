from __future__ import annotations

import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_objective_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    objective_range: dict,
):
    """Reset the objective position for each car."""
    num_envs = env.num_envs
    device = env.device

    # Get current car positions
    car_positions = env.scene["robot"].data.root_pos_w[:, :2]

    # Generate new objective positions
    x_offset = torch.rand(num_envs, device=device) * (objective_range["x"][1] - objective_range["x"][0]) + objective_range["x"][0]
    y_offset = torch.rand(num_envs, device=device) * (objective_range["y"][1] - objective_range["y"][0]) + objective_range["y"][0]
    new_positions = car_positions + torch.stack([x_offset, y_offset], dim=-1)
    env.objective_positions = new_positions

    # Create pose tensor: [x, y, z, qx, qy, qz, qw]
    poses = torch.zeros((num_envs, 7), device=device)
    poses[:, 0:2] = new_positions
    poses[:, 2] = 1.0  # z position
    poses[:, 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # identity rotation

    # Write poses
    env.scene["objective_cones"].write_root_pose_to_sim(poses)