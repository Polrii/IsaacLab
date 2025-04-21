from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

from . import observations as obs

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def distance_to_objective_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for being closer to the goal."""
    # Get the car positions
    car_positions = env.scene[asset_cfg.name].data.root_pos_w[:, :2] # (x, y)
    buffer_exists = hasattr(env, "previous_car_positions")

    # Initialize the buffer for previous positions if it doesn't exist
    if not buffer_exists:
        env.previous_car_positions = torch.zeros_like(car_positions, device=env.device)

    # Access the previous car positions
    previous_positions = env.previous_car_positions
    
    """ This part is used to update the previous positions in the buffer, but as we use the buffer on another reward,
    we will update them there
    # Update the buffer with the current positions
    env.previous_car_positions = car_positions.clone()"""
    
    # Get the objective positions
    objective_positions = env.scene["objective_cones"].data.root_pos_w[:, :2] # (x, y)
    
    # Get the starting positions
    starting_positions = env.scene["starting_cones"].data.root_pos_w[:, :2] # (x, y)
    
    # Compute distances
    current_distances = torch.norm(car_positions - objective_positions, dim=1)
    starting_distances = torch.norm(starting_positions - objective_positions, dim=1)
    if not buffer_exists:
        previous_distances = starting_distances
    else:
        previous_distances = torch.norm(previous_positions - objective_positions, dim=1)
    
    # Compute the rewards
    rewards = (current_distances - previous_distances) / (starting_distances + 1e-8)
    return rewards
    


def speed_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving faster."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get the objective positions
    objective_positions = env.scene["objective_cones"].data.root_pos_w[:, :2]  # (x, y)
    # Get the starting positions
    starting_positions = env.scene["starting_cones"].data.root_pos_w[:, :2]  # (x, y)
    
    # Constants
    max_speed = 285  # rad/s
    max_acceleration = 25  # rad/s^2
    max_linear_vel = 94  # m/s
    
    # Calculate frames needed
    acceleration_time = max_speed / max_acceleration
    acceleration_distance = 0.5 * acceleration_time**2
    
    # Compute distances
    distances = torch.norm(starting_positions - objective_positions, dim=1)  # Shape: [num_envs]
    
    # Use torch.where to handle the condition
    time_needed = torch.where(
        distances < acceleration_distance,
        torch.sqrt(2 * distances / max_acceleration),
        acceleration_time + ((distances - acceleration_distance) / max_acceleration)
    )
    
    frames_needed = time_needed / env.step_dt

    # Get the current linear velocity of the car
    velocity = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)  # xy-plane velocity
    
    # Calculate the speed reward
    rewards = velocity / (max_linear_vel * frames_needed)
    return rewards


def time_penalty(
    env: ManagerBasedRLEnv
) -> torch.Tensor:
    """Penalty for taking longer than needed."""
    # Get the objective positions
    objective_positions = env.scene["objective_cones"].data.root_pos_w[:, :2]  # (x, y)
    # Get the starting positions
    starting_positions = env.scene["starting_cones"].data.root_pos_w[:, :2]  # (x, y)

    # Constants
    max_speed = 285  # rad/s
    max_acceleration = 25  # rad/s^2
    
    # Calculate frames needed
    acceleration_time = max_speed / max_acceleration
    acceleration_distance = 0.5 * acceleration_time**2
    
    # Compute distances
    distances = torch.norm(starting_positions - objective_positions, dim=1)  # Shape: [num_envs]
    
    # Use torch.where to handle the condition
    time_needed = torch.where(
        distances < acceleration_distance,
        torch.sqrt(2 * distances / max_acceleration),
        acceleration_time + ((distances - acceleration_distance) / max_acceleration)
    )
    return env.step_dt / time_needed
    


def travelled_distance_penalty(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for travelling a distance."""
    # Get the car positions
    car_positions = env.scene[asset_cfg.name].data.root_pos_w[:, :2] # (x, y)
    buffer_exists = hasattr(env, "previous_car_positions")

    # Initialize the buffer for previous positions if it doesn't exist
    if not buffer_exists:
        env.previous_car_positions = torch.zeros_like(car_positions, device=env.device)

    # Access the previous car positions
    previous_positions = env.previous_car_positions
    
    # Update the buffer with the current positions
    env.previous_car_positions = car_positions.clone()
    
    # Get the objective positions
    objective_positions = env.scene["objective_cones"].data.root_pos_w[:, :2] # (x, y)
    
    # Get the starting positions
    starting_positions = env.scene["starting_cones"].data.root_pos_w[:, :2] # (x, y)
    
    # Compute rewards
    rewards = torch.norm(car_positions - previous_positions, dim=1) / torch.norm(objective_positions - starting_positions, dim=1)
    return rewards


def objective_reached_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for reaching the objective."""
    car_positions = env.scene[asset_cfg.name].data.root_pos_w[:, :2] # (x, y)
    objective_positions = env.scene["objective_cones"].data.root_pos_w[:, :2] # (x, y)
    distances = torch.norm(car_positions - objective_positions, dim=1)
    return (distances < threshold).float()












def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    return (up_proj > threshold).float()


def move_to_target_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving to the target heading."""
    heading_proj = obs.base_heading_proj(env, target_pos, asset_cfg).squeeze(-1)
    return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)


class progress_reward(ManagerTermBase):
    """Reward for making progress towards the target."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # initialize the base class
        super().__init__(cfg, env)
        # create history buffer
        self.potentials = torch.zeros(env.num_envs, device=env.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

    def reset(self, env_ids: torch.Tensor):
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = self._env.scene["robot"]
        # compute projection of current heading to desired heading vector
        target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)
        to_target_pos = target_pos - asset.data.root_pos_w[env_ids, :3]
        # reward terms
        self.potentials[env_ids] = -torch.norm(to_target_pos, p=2, dim=-1) / self._env.step_dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute vector to target
        target_pos = torch.tensor(target_pos, device=env.device)
        to_target_pos = target_pos - asset.data.root_pos_w[:, :3]
        to_target_pos[:, 2] = 0.0
        # update history buffer and compute new potential
        self.prev_potentials[:] = self.potentials[:]
        self.potentials[:] = -torch.norm(to_target_pos, p=2, dim=-1) / env.step_dt

        return self.potentials - self.prev_potentials


class joint_pos_limits_penalty_ratio(ManagerTermBase):
    """Penalty for violating joint position limits weighted by the gear ratio."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        if "asset_cfg" not in cfg.params:
            cfg.params["asset_cfg"] = SceneEntityCfg("robot")
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self, env: ManagerBasedRLEnv, threshold: float, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute the penalty over normalized joints
        joint_pos_scaled = math_utils.scale_transform(
            asset.data.joint_pos, asset.data.soft_joint_pos_limits[..., 0], asset.data.soft_joint_pos_limits[..., 1]
        )
        # scale the violation amount by the gear ratio
        violation_amount = (torch.abs(joint_pos_scaled) - threshold) / (1 - threshold)
        violation_amount = violation_amount * self.gear_ratio_scaled

        return torch.sum((torch.abs(joint_pos_scaled) > threshold) * violation_amount, dim=-1)


class power_consumption(ManagerTermBase):
    """Penalty for the power consumed by the actions to the environment.

    This is computed as commanded torque times the joint velocity.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        if "asset_cfg" not in cfg.params:
            cfg.params["asset_cfg"] = SceneEntityCfg("robot")
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(self, env: ManagerBasedRLEnv, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # return power = torque * velocity (here actions: joint torques)
        return torch.sum(torch.abs(env.action_manager.action * asset.data.joint_vel * self.gear_ratio_scaled), dim=-1)








def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)
