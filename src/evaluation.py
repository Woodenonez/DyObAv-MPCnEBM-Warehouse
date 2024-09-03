import math
import json
import statistics
from typing import Optional, TypedDict

import numpy as np
from shapely.geometry import Polygon, Point # type: ignore


class RobotEvaluationInfo(TypedDict):
    complete_results: list[bool]
    collision_results: list[bool]
    smoothness_results: list[list[float]] # [speed, angular_speed]
    clearance_results: list[float]
    clearance_dyn_results: list[float]
    deviation_results: list[list[float]] # [mean, max]
    solve_mpc_time_list: list[float]


class Evaluator:
    """Evaluator for robots in a repeat experiment."""
    def __init__(self, robot_ids:list, robot_size: float, human_size: float, repeat: int) -> None:
        self.robot_ids = robot_ids
        self.r_size = robot_size
        self.h_size = human_size
        self.repeat = repeat

        self.solve_mmp_time_list:list[float] = []
        self.eval_results = {robot_id: self._init_evaluation(repeat) for robot_id in robot_ids}

    def _init_evaluation(self, repeat:int=1) -> RobotEvaluationInfo:
        return RobotEvaluationInfo(
            collision_results=[False]*repeat,
            complete_results=[False]*repeat,
            smoothness_results=[[math.inf, math.inf]]*repeat,
            clearance_results=[math.inf]*repeat,
            clearance_dyn_results=[math.inf]*repeat,
            deviation_results=[[math.inf, math.inf]]*repeat,
            solve_mpc_time_list=[])

    def report(self, robot_id: int, full_report:bool=False, save_path:Optional[str]=None):
        collision_results = self.eval_results[robot_id]["collision_results"]
        complete_results = self.eval_results[robot_id]["complete_results"]
        smoothness_results = [x for x in self.eval_results[robot_id]["smoothness_results"] if x[0] != math.inf]
        clearance_results = [x for x in self.eval_results[robot_id]["clearance_results"] if x != math.inf]
        clearance_dyn_results = [x for x in self.eval_results[robot_id]["clearance_dyn_results"] if x != math.inf]
        deviation_results = [x for x in self.eval_results[robot_id]["deviation_results"] if x[0] != math.inf]
        solve_mmp_time_list = self.solve_mmp_time_list
        solve_mpc_time_list = self.eval_results[robot_id]["solve_mpc_time_list"]

        if save_path is not None: # save to json
            total_results = {
                "collision_results": collision_results,
                "complete_results": complete_results,
                "smoothness_results": smoothness_results,
                "clearance_results": clearance_results,
                "clearance_dyn_results": clearance_dyn_results,
                "deviation_results": deviation_results,
                "solve_mmp_time_list": solve_mmp_time_list,
                "solve_mpc_time_list": solve_mpc_time_list
            }
            with open(save_path, 'w') as f:
                json.dump(total_results, f)


        print('='*10, f'Robot {robot_id}', '='*10)
        print('MMP solve time mean/max:',
              round(np.mean(np.array(solve_mmp_time_list[5:])), 3), '/', # warm up
              round(np.amax(np.array(solve_mmp_time_list[5:])), 3))
        print('MPC solve time mean/max:', 
              round(np.mean(np.array(solve_mpc_time_list[5:])), 3), '/',
              round(np.amax(np.array(solve_mpc_time_list[5:])), 3))
        print('-'*30)
        # print('Collision results:', collision_results)
        print('Success rate:', sum(complete_results)/len(complete_results))
        print('Collision rate:', sum(collision_results)/len(collision_results))
        if full_report:
            print('-'*30)
            # print('Smoothness results:', smoothness_results)
            print('Smoothness mean:', np.mean(np.array(smoothness_results), axis=0))
            print('-'*30)
            # print('Clearance results:', clearance_results)
            print('Clearance mean:', round(np.mean(np.array(clearance_results)), 3))

            # print('Clearance results (dyn):', clearance_dyn_results)
            print('Clearance mean (dyn):', round(np.mean(np.array(clearance_dyn_results)), 3))
            print('-'*30)
            # print('Deviation results:', deviation_results)
            print('Deviation mean:', round(np.mean(np.array(deviation_results)), 3))
            print('Deviation std:', round(np.std(np.array(deviation_results)), 3))
            print('Deviation max:', round(np.amax(np.array(deviation_results)), 3) if len(deviation_results)>0 else np.inf)
        print('='*30)

    def append_mmp_solve_time(self, solve_time: float):
        self.solve_mmp_time_list.append(solve_time)

    def check_collision(self, robot_id, num_repeat: int, state: np.ndarray, 
                        other_states: Optional[list[np.ndarray]]=None,
                        static_obstacles: Optional[list[list[tuple]]]=None, 
                        dynamic_obstacle: Optional[list[tuple]]=None) -> bool:
        """Call this function at each time step."""
        collision = False
        pos = Point(state[0], state[1])
        if other_states is not None:
            for other_state in other_states:
                if pos.distance(Point(other_state[0], other_state[1])) <= (self.r_size + self.r_size):
                    collision = True
        if static_obstacles is not None:
            for sobs in static_obstacles:
                if Polygon(sobs).contains(pos):
                    collision = True
        if dynamic_obstacle is not None:
            for dobs in dynamic_obstacle:
                if pos.distance(Point(dobs[0], dobs[1])) <= (self.h_size + self.r_size):
                    collision = True
        self.eval_results[robot_id]["collision_results"][num_repeat] = collision
        return collision

    def check_completion(self, robot_id: int, num_repeat: int, if_complete: bool):
        """Call this function at the end of each episode."""
        self.eval_results[robot_id]["complete_results"][num_repeat] = if_complete

    def append_tracker_solve_time(self, robot_id: int, solve_time: float):
        self.eval_results[robot_id]["solve_mpc_time_list"].append(solve_time)

    def calc_action_smoothness(self, robot_id: int, num_repeat: int, action_list: list[np.ndarray]):
        speeds = np.array(action_list)[:, 0]
        angular_speeds = np.array(action_list)[:, 1]
        smoothness = [statistics.mean(np.abs(np.diff(speeds, n=2))), statistics.mean(np.abs(np.diff(angular_speeds, n=2)))]
        self.eval_results[robot_id]["smoothness_results"][num_repeat] = smoothness

    def calc_minimal_obstacle_distance(self, robot_id: int, num_repeat: int, trajectory: list[tuple], obstacles: list[list[tuple]]):
        dist_list = []
        for pos in trajectory:
            dist_list.append(min([Polygon(obs).distance(Point(pos)) for obs in obstacles]))
        self.eval_results[robot_id]["clearance_results"][num_repeat] = min(dist_list)

    def calc_minimal_dynamic_obstacle_distance(self, robot_id: int, num_repeat: int, state: np.ndarray, obstacles: list[tuple]):
        """Call this function at each time step."""
        dist_list = [np.linalg.norm(state[:2]-np.array(obstacle)) for obstacle in obstacles]
        if float(min(dist_list)) < self.eval_results[robot_id]["clearance_dyn_results"][num_repeat]:
            self.eval_results[robot_id]["clearance_dyn_results"][num_repeat] = float(min(dist_list))

    def calc_deviation_distance(self, robot_id: int, num_repeat: int, ref_traj: list[tuple], actual_traj: list[tuple]):
        deviation_dists = []
        for pos in actual_traj:
            deviation_dists.append(min([math.hypot(ref_pos[0]-pos[0], ref_pos[1]-pos[1]) for ref_pos in ref_traj]))
        self.eval_results[robot_id]["deviation_results"][num_repeat] = [statistics.mean(deviation_dists), max(deviation_dists)]






