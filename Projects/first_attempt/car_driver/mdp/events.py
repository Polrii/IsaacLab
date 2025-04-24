from __future__ import annotations

import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter


def reset_car_pos_and_other_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    starting_pos_asset_cfg: SceneEntityCfg = SceneEntityCfg("starting_cones"),
):
    """Reset the car position and other positions for specific environments."""
    num_envs = env.num_envs
    device = env.device
    
    """Reset the car"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    
    
    """Reset the starting position cones"""
    # Get the cone objects
    cone_asset: RigidObject = env.scene[starting_pos_asset_cfg.name]
    
    # Build a full pose tensor (for all envs)
    cone_poses = cone_asset.data.root_pos_w.clone()  # shape: (num_envs, 3)

    # Create identity rotations (same shape as orientation, (num_envs, 4))
    identity_quats = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(num_envs, 1)

    # Concatenate positions + rotations => shape (num_envs, 7)
    full_poses = torch.cat([cone_poses, identity_quats], dim=-1)

    # Now modify only selected env_ids
    full_poses[env_ids, 0:2] = positions[:, 0:2].clone()  # set x and y to the new positions
    full_poses[env_ids, 2] = torch.full((len(env_ids),), 3.0, device=device) # set z to 3.0
    # Apply only to selected environments
    cone_asset.write_root_pose_to_sim(full_poses[env_ids], env_ids=env_ids)

    
    """Reset the previous positions buffer"""
    # Get current car positions
    car_positions = positions[:, :2]  # (x, y)
    buffer_exists = hasattr(env, "previous_car_positions")

    # Initialize the buffer for previous positions if it doesn't exist
    if not buffer_exists:
        env.previous_car_positions = torch.zeros_like(car_positions, device=env.device)
    # Store the previous positions
    env.previous_car_positions[env_ids] = car_positions



def reset_objective_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict,
):
    """Reset the objective position for specific environments."""
    num_envs = env.num_envs
    device = env.device

    # Get environment origins
    env_origins = env.scene.env_origins[:, :2]

    # Generate random offsets
    x_offset = torch.rand(len(env_ids), device=device) * (position_range["x"][1] - position_range["x"][0]) + position_range["x"][0]
    y_offset = torch.rand(len(env_ids), device=device) * (position_range["y"][1] - position_range["y"][0]) + position_range["y"][0]
    new_positions = env_origins[env_ids] + torch.stack([x_offset, y_offset], dim=-1)

    # Store updated objective positions
    if not hasattr(env, "objective_positions"):
        env.objective_positions = torch.zeros((num_envs, 2), device=device)
    env.objective_positions[env_ids] = new_positions

    # Create poses tensor using current root position and rotation
    pos = env.scene["objective_cones"].data.root_pos_w.clone()
    # Use identity quaternion (no rotation)
    identity_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).repeat(pos.shape[0], 1)
    poses = torch.cat([pos, identity_rot], dim=-1)

    # Update only the ones in env_ids
    poses[env_ids, 0:2] = new_positions
    poses[env_ids, 2] = 1.0  # z height
    poses[env_ids, 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)

    # Apply only updated poses
    env.scene["objective_cones"].write_root_pose_to_sim(poses)