import os
import pathlib

from configs import MpcConfiguration, CircularRobotSpecification

from basic_motion_model import motion_model
from pkg_tracker_mpc.casadi_build import builder_panoc

def return_cfg_path(fname: str) -> str:
    root_dir = pathlib.Path(__file__).resolve().parents[1]
    cfg_path = os.path.join(root_dir, "config", fname)
    return cfg_path

def load_mpc_config(fname: str) -> MpcConfiguration:
    """Load the MPC configuration."""
    return MpcConfiguration.from_yaml(return_cfg_path(fname))

def load_robot_spec(fname: str) -> CircularRobotSpecification:
    """Load the robot specification."""
    return CircularRobotSpecification.from_yaml(return_cfg_path(fname))

if __name__ == "__main__":
    cfg_fname = "mpc_fast.yaml"
    # cfg_fname = "mpc_default.yaml"

    robot_spec = "spec_robot.yaml"
    config_mpc = load_mpc_config(cfg_fname)
    config_robot = load_robot_spec(robot_spec)
    mpc_module = builder_panoc.PanocBuilder(config_mpc, config_robot)
    mpc_module.load_motion_model(motion_model.unicycle_model)
    mpc_module.build(test=False)