"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Example project where an AI controls a car.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
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
    
    # Object origins
    origins = [[-0.835, -1.0, 0.33], [0.835, -1.0, 0.33], [-0.86, -3.72, 0.33], [0.86, -3.72, 0.33], [0.0, -2.25, 0.725]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin) 
    
    # Configs
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
    
    # Create wheel and body objects
    fl_wheel_cfg = front_wheel_cfg.copy()
    fl_wheel_cfg.prim_path = "/World/Origin0/FrontLeftWheel"
    fl_wheel = RigidObject(cfg=fl_wheel_cfg)

    fr_wheel_cfg = front_wheel_cfg.copy()
    fr_wheel_cfg.prim_path = "/World/Origin1/FrontRightWheel"
    fr_wheel = RigidObject(cfg=fr_wheel_cfg)

    bl_wheel_cfg = back_wheel_cfg.copy()
    bl_wheel_cfg.prim_path = "/World/Origin2/BackLeftWheel"
    bl_wheel = RigidObject(cfg=bl_wheel_cfg)

    br_wheel_cfg = back_wheel_cfg.copy()
    br_wheel_cfg.prim_path = "/World/Origin3/BackRightWheel"
    br_wheel = RigidObject(cfg=br_wheel_cfg)

    car_body_cfg_copy = car_body_cfg.copy()
    car_body_cfg_copy.prim_path = "/World/Origin4/Body"
    car_body = RigidObject(cfg=car_body_cfg_copy)

    simulation_app.update()  # Allow stage to update with new prims

    # Setup articulation: Make car body the articulation root
    dc.create_articulation("/World/Origin4/Body")
    root_handle = dc.get_articulation("/World/Origin4/Body")
    car_link = dc.get_articulation_root_body(root_handle)

    def create_wheel_joint(wheel_prim_path: str, joint_name: str):
        wheel_handle = dc.get_rigid_body(wheel_prim_path)
        joint_handle = dc.create_joint(
            parent=car_link,
            child=wheel_handle,
            joint_type=dc.JointType.REVOLUTE,
            axis=[1.0, 0.0, 0.0],
            parent_pose=[0, 0, 0, 1, 0, 0, 0],
            child_pose=[0, 0, 0, 1, 0, 0, 0],
            name=joint_name
        )
        dc.set_joint_drive_mode(joint_handle, dc.JointDriveMode.VELOCITY)
        dc.set_joint_target_velocity(joint_handle, 0.0)
        dc.set_joint_max_force(joint_handle, 5000.0)

    # Create joints for each wheel
    create_wheel_joint("/World/Origin0/FrontLeftWheel", "fl_joint")
    create_wheel_joint("/World/Origin1/FrontRightWheel", "fr_joint")
    create_wheel_joint("/World/Origin2/BackLeftWheel", "bl_joint")
    create_wheel_joint("/World/Origin3/BackRightWheel", "br_joint")

    scene = {
        "fl_wheel": fl_wheel,
        "fr_wheel": fr_wheel,
        "bl_wheel": bl_wheel,
        "br_wheel": br_wheel,
        "car_body": car_body,
    }
    return scene, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject], origins: torch.Tensor):
    """Runs the simulation loop."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    while simulation_app.is_running():
        for obj in entities.values():
            obj.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        for obj in entities.values():
            obj.update(sim_dt)


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[10.0, 0.0, 10.0], target=[0.0, 0.0, 0.0])
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
