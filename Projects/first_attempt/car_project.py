"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Example project where an AI controls a car.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass


"""Config for the articulation"""
# We keep this outside @configclass to avoid issue because there is no prim_path
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


"""Designs the scene."""
@configclass
class CarSceneCfg(InteractiveSceneCfg):
    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # Lights
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight", spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    )
    
    # Articulation
    car: ArticulationCfg = car_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability
    robot = scene["car"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    max_angular_velocity = 285.0 # Rad/s
    
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            
            # Get joint velocities
            joint_vel = robot.data.default_joint_vel.clone()
            
            # Make a copy of joint_vel for a target
            joint_vel_target = joint_vel.clone()
            
            # Assign random angular velocities to each wheel in each environment
            for env_idx in range(scene.num_envs):
                for wheel in ["FrontLeft", "FrontRight", "BackLeft", "BackRight"]:
                    joint_name_idx = robot.data.joint_names.index(wheel)
                    joint_vel_target[env_idx, joint_name_idx] = torch.rand(1).item() * 2 * max_angular_velocity - max_angular_velocity
            
            print(f"[DATA]: Joint velocity targets: {joint_vel_target}")
            robot.set_joint_velocity_target(joint_vel_target)

            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        
        """
        # Get the joint velocities
        current_joint_velocities = robot.data.joint_vel.clone()
        print(f"[DATA]: Current joint velocities: {current_joint_velocities}") # Rad/s
        """
        
        # Wite data to sim
        scene.write_data_to_sim()
            
        # Perform step
        sim.step()
        
        # Increment counter
        count += 1
        
        # Update buffers
        scene.update(sim_dt)

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[40.0, 0.0, 50.0], target=[0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = CarSceneCfg(num_envs=args_cli.num_envs, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()