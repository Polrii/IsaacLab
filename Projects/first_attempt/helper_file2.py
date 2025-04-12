"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# create argparser
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
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg, ArticulationCfg, Articulation
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationContext

from omni.isaac.dynamic_control import _dynamic_control
dc = _dynamic_control.acquire_dynamic_control_interface()


def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # Lights
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))
    cfg_light_dome = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg_light_dome.func("/World/Light", cfg_light_dome)
    
    # Create separate groups for the different objects
    origins = [[-0.835, -1.0, 0.33], [0.835, -1.0, 0.33], [-0.86, -3.72, 0.33], [0.86, -3.72, 0.33], [0.0, -2.25, 0.725], [0.0, 0.0, 0.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin) 
        
    # Create the configuration for the articulation
    car_cfg = ArticulationCfg(
        prim_path="/World/Origin5/Car",
        spawn=sim_utils.CylinderCfg(
            radius=0.33,
            height=0.25,
            axis="X",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=94.0,
                max_angular_velocity=16320.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=18.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.8)),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), joint_pos={"slider_to_cart": 0.0, "cart_to_pole": 0.0}
        ),
        actuators={
            "front_left_wheel_actuator": ImplicitActuatorCfg(
                joint_names_expr=["fl_joint"],
                effort_limit=400.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=10.0,
            ),
        },
    )
    
    # Create the car articulation
    car_object = Articulation(cfg=car_cfg)
    
    """
    # Rigid Cylinder Configuration for front wheels
    front_wheel_cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/FrontWheel",
        spawn=sim_utils.CylinderCfg(
            radius=0.33,
            height=0.25,
            axis="X",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=18.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    
    # Rigid Cylinder Configuration for back wheels
    back_wheel_cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/BackWheel",
        spawn=sim_utils.CylinderCfg(
            radius=0.33,
            height=0.3,
            axis="X",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=21.6),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.4, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    
    # Rigid Rectangle Configuration for the car body
    car_body_cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/CarBody",
        spawn=sim_utils.CuboidCfg(
            size=[1.4, 4.5, 1.25],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1620.8),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    
    # Create the front wheels
    fl_wheel_cfg = front_wheel_cfg.copy()
    fl_wheel_cfg.prim_path = "/World/Origin0/FrontLeftWheel"
    fl_wheel_object = RigidObject(cfg=fl_wheel_cfg)

    fr_wheel_cfg = front_wheel_cfg.copy()
    fr_wheel_cfg.prim_path = "/World/Origin1/FrontRightWheel"
    fr_wheel_object = RigidObject(cfg=fr_wheel_cfg)

    # Create the back wheels
    bl_wheel_cfg = back_wheel_cfg.copy()
    bl_wheel_cfg.prim_path = "/World/Origin2/BackLeftWheel"
    bl_wheel_object = RigidObject(cfg=bl_wheel_cfg)

    br_wheel_cfg = back_wheel_cfg.copy()
    br_wheel_cfg.prim_path = "/World/Origin3/BackRightWheel"
    br_wheel_object = RigidObject(cfg=br_wheel_cfg)

    # Create the car body
    car_body_cfg_copy = car_body_cfg.copy()
    car_body_cfg_copy.prim_path = "/World/Origin4/Body"
    car_body_object = RigidObject(cfg=car_body_cfg_copy)
    
    simulation_app.update()  # Allow stage to update with new prims

    # Get the car body as a rigid body for joint creation
    car_link = dc.get_rigid_body("/World/Origin4/Body")
    
    def create_wheel_joint(wheel_prim_path: str, joint_name: str):
        wheel_handle = dc.get_rigid_body(wheel_prim_path)

        joint_handle = dc.create_d6_joint(
            parent=car_link,
            child=wheel_handle,
            joint_pose_in_parent=_dynamic_control.Transform(),
            joint_pose_in_child=_dynamic_control.Transform()
        )

        # Lock all axes except X (rotation)
        dc.set_dof_locked(joint_handle, 0, False)  # X rotation (allow spin)
        dc.set_dof_locked(joint_handle, 1, True)   # Y rotation
        dc.set_dof_locked(joint_handle, 2, True)   # Z rotation
        dc.set_dof_locked(joint_handle, 3, True)   # X translation
        dc.set_dof_locked(joint_handle, 4, True)   # Y translation
        dc.set_dof_locked(joint_handle, 5, True)   # Z translation

        # Set drive mode, velocity, and force for the wheel
        dc.set_dof_drive_mode(joint_handle, 0, dc.DofDriveMode.VELOCITY)
        dc.set_dof_target_velocity(joint_handle, 0, 0.0)
        dc.set_dof_max_force(joint_handle, 0, 1000.0)
    
    # Create joints for each wheel
    create_wheel_joint("/World/Origin0/FrontLeftWheel", "fl_joint")
    create_wheel_joint("/World/Origin1/FrontRightWheel", "fr_joint")
    create_wheel_joint("/World/Origin2/BackLeftWheel", "bl_joint")
    create_wheel_joint("/World/Origin3/BackRightWheel", "br_joint")

    
    scene_entities = {
        "fl_wheel": fl_wheel_object,
        "fr_wheel": fr_wheel_object,
        "bl_wheel": bl_wheel_object,
        "br_wheel": br_wheel_object,
        "car_body": car_body_object,
    }
    """
    scene_entities = {"car": car_object}
    # Return the scene information
    # Set the initial state of the wheels and car body
    return scene_entities, origins



def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    # Simulate physics
    while simulation_app.is_running():
        # apply sim data
        for object in entities.values():
            object.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        # update buffers
        for object in entities.values():
            object.update(sim_dt)

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
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