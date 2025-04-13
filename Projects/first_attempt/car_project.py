"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Example project where an AI controls a car.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext
from isaaclab.actuators import ImplicitActuatorCfg


"""Designs the scene."""
def design_scene() -> tuple[dict, list[list[float]]]:
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # Lights
    cfg_light_distant = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))
    cfg_light_dome = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg_light_dome.func("/World/Light", cfg_light_dome)

    # Create separate groups called "Origin1", "Origin2"
    # Each group will have a robot in it
    origins = [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]]
    # Origin 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # Origin 2
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    # Create the configuration for the car articulation
    car_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"C:/Users/mcpek/IsaacLab/Projects/first_attempt/Car_object.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=16320.0,
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
                effort_limit=1200.0,
                velocity_limit=None,
                stiffness=0.0,
                damping=10.0,
            ),
            "front_right_actuator": ImplicitActuatorCfg(
                joint_names_expr=["FrontRight"],
                effort_limit=1200.0,
                velocity_limit=None,
                stiffness=0.0,
                damping=10.0,
            ),
            "back_left_actuator": ImplicitActuatorCfg(
                joint_names_expr=["BackLeft"],
                effort_limit=1200.0,
                velocity_limit=None,
                stiffness=0.0,
                damping=10.0,
            ),
            "back_right_actuator": ImplicitActuatorCfg(
                joint_names_expr=["BackRight"],
                effort_limit=1200.0,
                velocity_limit=None,
                stiffness=0.0,
                damping=10.0,
            ),
        },
    )
    
    # Articulation
    car_cfg_copy = car_cfg.copy()
    car_cfg_copy.prim_path = "/World/Origin.*/Robot"
    car_object = Articulation(cfg=car_cfg_copy)

    # return the scene information
    scene_entities = {"car": car_object}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["car"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
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
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_vel += torch.rand_like(joint_vel) * 1000.0
            print(f"[DATA]: Joint pos: {joint_pos}, Joint vel: {joint_vel}")
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")
        # Apply random action
        # -- generate random joint efforts
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- apply action to the robot
        robot.set_joint_effort_target(efforts)
        
        # -- write data to sim
        for object in entities.values():
            object.write_data_to_sim()
            
        # Perform step
        sim.step()
        
        # Increment counter
        count += 1
        
        # Update buffers
        for object in entities.values():
            object.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[10.0, 0.0, 10.0], target=[0.0, 0.0, 0.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()