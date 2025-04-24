from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import sys
sys.path.append("C:/Users/mcpek/IsaacLab/Projects/first_attempt")
import car_driver.mdp as mdp



def objective_reached(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    threshold: float = 1.0
) -> torch.Tensor:
    """Terminate when the car reaches the objective."""
    # Get the car positions
    car_positions = env.scene[asset_cfg.name].data.root_pos_w[:, :2] # (x, y)
    
    # Get the objective positions
    objective_positions = env.scene["objective_cones"].data.root_pos_w[:, :2] # (x, y)
    
    # compute distance to objective
    distances = torch.norm(car_positions - objective_positions, dim=1)
    mask = distances < threshold
    
    if mask.any():
        # Apply the objective reached bonus
        env.reward_buf[mask] += (mdp.objective_reached_bonus(env, threshold=threshold)[mask]*1000000)
        
        print(f"[Termination]: Terminated because objective was reached! Distances below threshold: {distances[mask]}")
    
    return mask