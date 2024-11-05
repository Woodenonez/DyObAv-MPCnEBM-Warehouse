import os
import json
import pathlib

import numpy as np

from basic_motion_model.motion_model import UnicycleModel
from basic_map.map_geometric import GeometricMap

from pkg_motion_plan.global_path_coordinate import GlobalPathCoordinator
from pkg_motion_plan.local_traj_plan import LocalTrajPlanner
from pkg_tracker_mpc.trajectory_tracker import TrajectoryTracker
from pkg_robot.robot import RobotManager

from configs import MpcConfiguration
from configs import CircularRobotSpecification

from visualizer.object import CircularObjectVisualizer
from visualizer.mpc_plot import MpcPlotInLoop # type: ignore

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = os.path.join(ROOT_DIR, "data", "test_vicon")
CNFG_DIR = os.path.join(ROOT_DIR, "config")
VB = False
TIMEOUT = 1000

robot_ids = None # if none, read from schedule

### Configurations
config_mpc_path = os.path.join(CNFG_DIR, "mpc_default.yaml")
config_robot_path = os.path.join(CNFG_DIR, "spec_robot.yaml")

config_mpc = MpcConfiguration.from_yaml(config_mpc_path)
config_robot = CircularRobotSpecification.from_yaml(config_robot_path)

### Map, paths
map_path = os.path.join(DATA_DIR, "map.json")
geo_map = GeometricMap.from_json(map_path)
inflated_map = GlobalPathCoordinator.inflate_map(geo_map, config_robot.vehicle_width+0.2)

robot_start = (-2.0, 0.5, 0.0)
robot_end = (3.0, 0.5)

### Set up robots
robot_manager = RobotManager()

robot = robot_manager.create_robot(config_robot, UnicycleModel(sampling_time=config_mpc.ts), id_=0)
robot.set_state(np.asarray(robot_start))
planner = LocalTrajPlanner(config_mpc.ts, config_mpc.N_hor, config_robot.lin_vel_max, verbose=VB)
planner.load_map(inflated_map.boundary_coords, inflated_map.obstacle_coords_list)
controller = TrajectoryTracker(config_mpc, config_robot, robot_id=0, verbose=VB)
controller.load_motion_model(UnicycleModel(sampling_time=config_mpc.ts))
visualizer = CircularObjectVisualizer(config_robot.vehicle_width, indicate_angle=True)
robot_manager.add_robot(robot, controller, planner, visualizer)

robot_manager.add_schedule(0, np.asarray(robot_start), scheduled_path_coords=[robot_start, robot_end], scheduled_path_times=None)

### Run
main_plotter = MpcPlotInLoop(config_robot)
main_plotter.plot_in_loop_pre(geo_map, inflated_map, geo_map)

main_plotter.add_object_to_pre(0,
                               planner.ref_traj,
                               controller.state, # type: ignore
                               controller.final_goal, # type: ignore
                               color='b')
visualizer.plot(main_plotter.map_ax, *robot.state)

for kt in range(TIMEOUT):
    robot_states = []
    incomplete = False

    ref_states, ref_speed, *_ = planner.get_local_ref(kt*config_mpc.ts, (float(robot.state[0]), float(robot.state[1])) )
    if ref_speed is not None:
        print(f"Robot {0} ref speed: {round(ref_speed, 4)}") # XXX
    controller.set_ref_states(ref_states, ref_speed=ref_speed)
    actions, pred_states, current_refs, debug_info = controller.run_step(static_obstacles=inflated_map.obstacle_coords_list,
                                                                         full_dyn_obstacle_list=None,
                                                                         other_robot_states=None,
                                                                         map_updated=False)

    ### Real run
    robot.step(actions[-1])
    robot_manager.set_pred_states(0, np.asarray(pred_states))

    main_plotter.update_plot(0, kt, actions[-1], robot.state, debug_info['cost'], np.asarray(pred_states), current_refs)
    visualizer.update(*robot.state)

    if not controller.check_termination_condition(external_check=planner.idle):
        incomplete = True

    robot_states.append(robot.state)
    
    main_plotter.plot_in_loop(time=kt*config_mpc.ts, autorun=False, zoom_in=None)
    if not incomplete:
        break
    
    
main_plotter.show()
input('Press anything to finish!')
main_plotter.close()
