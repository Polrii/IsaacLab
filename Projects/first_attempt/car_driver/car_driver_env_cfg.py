import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import sys
sys.path.append("C:/Users/mcpek/IsaacLab/Projects/first_attempt")
import car_driver.mdp as mdp

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass

##
# Car config
##
car_cfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"C:/Users/mcpek/IsaacLab/Projects/first_attempt/Car_object.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=16320.0, # Deg/s
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0), joint_pos={"FrontLeft": 0.0, "FrontRight": 0.0, "BackLeft": 0.0, "BackRight": 0.0}
    ),
    actuators={
        "front_left_actuator": ImplicitActuatorCfg(
            joint_names_expr=["FrontLeft"],
            effort_limit_sim=1200.0,
            velocity_limit=None,
            stiffness=0.0,
            damping=100.0,
        ),
        "front_right_actuator": ImplicitActuatorCfg(
            joint_names_expr=["FrontRight"],
            effort_limit_sim=1200.0,
            velocity_limit=None,
            stiffness=0.0,
            damping=100.0,
        ),
        "back_left_actuator": ImplicitActuatorCfg(
            joint_names_expr=["BackLeft"],
            effort_limit_sim=1200.0,
            velocity_limit=None,
            stiffness=0.0,
            damping=100.0,
        ),
        "back_right_actuator": ImplicitActuatorCfg(
            joint_names_expr=["BackRight"],
            effort_limit_sim=1200.0,
            velocity_limit=None,
            stiffness=0.0,
            damping=100.0,
        ),
    },
)



##
# Scene definition
##


@configclass
class CarDriverSceneCfg(InteractiveSceneCfg):
    """Configuration for a car driver scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg(size=(10000, 10000)))

    # Lights
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight", spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    )
    
    # Car articulation
    robot: ArticulationCfg = car_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    joint_velocity = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["FrontLeft", "FrontRight", "BackLeft", "BackRight"], scale=100.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_vel = ObsTerm(func=mdp.joint_vel)           # velocity of each wheel
        root_lin_vel_w = ObsTerm(func=mdp.root_lin_vel_w) # linear velocity of the car
        root_ang_vel_w = ObsTerm(func=mdp.root_ang_vel_w) # angular velocity of the car
        root_pos = ObsTerm(func=mdp.root_pos_w)             # position of the car
        root_quat_w = ObsTerm(func=mdp.root_quat_w)       # quaternion of the car (orientation)
        base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (0.0, 0.0, 0.0)})



        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_car_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-200, 200), "y": (-200, 200)}, 
            "velocity_range": {"yaw": (-math.pi, math.pi)},
            },
    )
    
    reset_car_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0, 0), 
            "velocity_range": (0, 0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    # (1) Reward for moving forward
    progress = RewTerm(
        func=mdp.progress_reward, 
        weight=1.0, 
        params={"target_pos": (0.0, 0.0, 0.0)}
    )
    # (2) Reward for moving fast
    speed = RewTerm(
        func=mdp.speed_reward, 
        weight=0.1, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["FrontLeft", "FrontRight", "BackLeft", "BackRight"])}
    )
    # (3) Constant time penalty
    time_penalty = RewTerm(
        func=mdp.is_alive,
        weight=-0.01,
    )
    # (4) Travelled distance penalty
    travel_penalty = RewTerm(
        func=mdp.travelled_distance_penalty,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["FrontLeft", "FrontRight", "BackLeft", "BackRight"])}
    )
    # (5) Accomplishement reward
    terminating = RewTerm(func=mdp.is_terminated, weight=2.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class CarDriverEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the car driver environment."""

    # Scene settings
    scene: CarDriverSceneCfg = CarDriverSceneCfg(num_envs=4096, env_spacing=10.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 15.0 # episode length in seconds
        # viewer settings
        self.viewer.eye = (40.0, 0.0, 50.0)
        self.viewer.target = (0.0, 0.0, 0.0)
        # simulation settings
        self.sim.dt = 1 / 240
        self.sim.render_interval = self.decimation
