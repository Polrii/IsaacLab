from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm




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
    
    return distances < threshold