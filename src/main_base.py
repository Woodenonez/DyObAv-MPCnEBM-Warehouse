import math
import time
import pathlib
import warnings
import timeit
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes # type: ignore

from configs import MpcConfiguration, RPPConfiguration
from configs import TebConfiguration
from configs import CircularRobotSpecification
from configs import PedestrianSpecification
from configs import WarehouseSimConfiguration

from basic_map.map_tf import ScaleOffsetReverseTransform

from pkg_motion_plan.global_path_coordinate import GlobalPathCoordinator
from pkg_motion_plan.local_traj_plan import LocalTrajPlanner
from pkg_robot.robot import RobotManager
from pkg_moving_object.moving_object import HumanObject, RobotObject

from pkg_tracker_mpc.trajectory_tracker import TrajectoryTracker as TrajectoryTrackerMPC
from pkg_tracker_rpp.trajectory_tracker import TrajectoryTracker as TrajectoryTrackerRPP

from pkg_planner_teb.trajectory_planner import TrajectoryPlanner as TrajectoryPlannerTEB


from pkg_mp_ebm.motion_predict import MotionPredictor as MotionPredictorEBM
from pkg_mp_cvm.motion_predict import MotionPredictor as MotionPredictorCVM
from pkg_mp_sgan.motion_predict import MotionPredictor as MotionPredictorSGAN

from visualizer.object import CircularObjectVisualizer
from visualizer.mpc_plot import MpcPlotInLoop # type: ignore

from evaluation import Evaluator
# from draft import threat_assessment

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def scenario(index):
    """Specific scenarios for evaluation.

    Returns:
        human_starts: list of np.array, the start points of the human(s)
        human_paths: list of list of int, the path nodes of the human(s)
        robot_start_points: list[np.array], the start points of the robots
        robot_path: list[list[int]], the paths of the robots

    Notes:
        *Scenario 1 (1 robot): Abrupt appearance (of 1 pedestrian).
        *Scenario 2 (1 robot): Overtaken (by 1 pedestrian).
        Scenario 3 (2 robot): Dynamic narrow corridor (with 4 pedestrians).
        Scenario 4 (1 robot): Group avoidance (with 3 pedestrians).
        Scenario 10 (1 robot): L-shaped corridor encounter (between the robot and a pedestrian).
        Scenario 11 (1 robot): Pesdestrian turning (with a pedestrian).
        Scenario 12 (1 robot): Face-to-face encounter (with a pedestrian).
        Scenario 20 (5 robot): Long term.
    """
    if index == 1:
        human_starts = [np.array([110, 20, 0.0])]  
        human_paths = [[1, 2, 9, 32]]
        robot_start_points = [np.array([160, 160, math.pi/2])] # [1.0, -1.7, 1.571]
        robot_path = [[12, 11, 10, 9, 8]]
    elif index == 2:
        human_starts = [np.array([235, 0, 0.0])] 
        human_paths = [[15, 16, 27]]
        robot_start_points = [np.array([255, 20, -math.pi/2])]
        robot_path = [[20, 21, 22]]
    elif index == 3:
        human_starts = [np.array([160, 120, 0.0]),
                        np.array([160, 120, 0.0]),
                        np.array([300, 115, 0.0]),
                        np.array([300, 115, 0.0]),] 
        human_paths = [[32, 16, 27],
                       [32, 16, 22, 23],
                       [27, 16, 32],
                       [27, 16, 15, 20]]
        robot_start_points = [np.array([235, 195, math.pi/2]),
                              np.array([235, 20, -math.pi/2]),]
        robot_path = [[17, 15],
                      [15, 17],]
    elif index == 4:
        human_starts = [np.array([228, 20, 0.0]),
                        np.array([235, 18, 0.0]),
                        np.array([242, 20, 0.0])]
        human_paths = [[15, 19],
                       [15, 19],
                       [15, 19]]
        robot_start_points = [np.array([235, 120, math.pi/2])]
        robot_path = [[16, 15]]
    elif index == 10:
        human_starts = [np.array([160, 50, 0.0])]  
        human_paths = [[9, 32, 16]]
        robot_start_points = [np.array([245, 115, -math.pi]) ]
        robot_path = [[16, 32, 9]]
    elif index == 11:    
        human_starts = [np.array([160, 50, 0.0])]  
        human_paths = [[9, 32, 16]]
        robot_start_points = [np.array([160, 160, math.pi/2])] # [1.0, -1.7, 1.571]
        robot_path = [[12, 11, 10, 9, 8]]
    elif index == 12:
        human_starts = [np.array([160, 75, 0.0])]  
        human_paths = [[9, 12]]
        robot_start_points = [np.array([160, 165, math.pi/2])] # [1.0, -1.7, 1.571]
        robot_path = [[12, 9]]
    elif index == 13: # Face-to-face encounter for 2 robots
        human_starts = [np.array([110, 20, 0.0])]  
        human_paths = [[1, 2]]
        robot_start_points = [np.array([160, 165, math.pi/2]), np.array([160, 75, -math.pi/2])]
        robot_path = [[12, 9], [9, 12]]
    elif index == 20:
        human_starts = [
            np.array([110, 20, 0.0]),
            np.array([160, 20, 0.0]),
            np.array([235, 20, 0.0])
        ]
        human_paths = [
            [1, 4, 11, 32, 16, 15, 1, 4, 11, 32, 16, 15, 1, 4, 11, 32, 16, 15, 1],
            [8, 1, 2, 9, 10, 3, 4, 11, 10, 3, 2, 9, 8, 1, 2, 9, 10, 3, 4, 11, 10, 3, 2, 9, 8, 1],
            [15, 8, 32, 16, 15, 8, 32, 16, 15, 8, 32, 16, 15, 8, 32, 16, 15, 8, 1]
        ]
        robot_start_points = [
            np.array([235, 120, math.pi]),
            np.array([110, 80, math.pi/2]),
            np.array([110, 90, math.pi/2]),
        ]
        robot_path = [
            [16, 32, 8, 15, 16, 32, 8, 15, 16],
            [2, 1, 8, 11, 4, 1, 8, 11, 4, 2],
            [3, 2, 1, 8, 11, 4, 1, 8, 11, 4, 3],
        ]
                      
    else:
        raise ValueError('Invalid scenario index: %d' % index)
    return human_starts, human_paths, robot_start_points, robot_path


class MainBase:
    color_list = ['#2878b5', '#9ac9db', '#f8ac8c', '#c82423', '#bb9727', 
                  '#54b345', '#32b897', '#05b9e2', '#8983bf', '#c76da2',
                  '#f8ac8c', '#c82423', '#bb9727', '#54b345', '#32b897',]
    
    def __init__(self, scenario_index: int, evaluation=False, visualization=True, verbose=True):
        self._human_starts_img, self.human_paths, self._robot_starts_img, self.robot_paths = scenario(scenario_index)

        self.eval = False
        if evaluation:
            self.eval = True

        self.viz = visualization
        self.vb = verbose

    def _load_config(self, config_planner_path, config_tracker_path, config_robot_path, config_human_path):
        if self.vb:
            print(f'[{self.__class__.__name__}] Loading configuration...')
        if self.tracker_type == 'mpc':
            self.config_tracker = MpcConfiguration.from_yaml(config_tracker_path)
        elif self.tracker_type == 'rpp':
            self.config_tracker = RPPConfiguration.from_yaml(config_tracker_path)
            
        if self.planner_type == 'teb':
            self.config_planner = TebConfiguration.from_yaml(config_planner_path)
        self.config_robot = CircularRobotSpecification.from_yaml(config_robot_path)
        self.config_human = PedestrianSpecification.from_yaml(config_human_path)

    def _load_tf(self, config_tf_path):
        if self.vb:
            print(f'[{self.__class__.__name__}] Loading frame transformer...')
        self.config_tf = WarehouseSimConfiguration.from_yaml(config_tf_path)
        self.tf_img2real = ScaleOffsetReverseTransform(
            scale=self.config_tf.scale2real, 
            offsetx_after=self.config_tf.corner_coords[0], 
            offsety_after=self.config_tf.corner_coords[1], 
            y_reverse=(not self.config_tf.image_axis), 
            y_max_before=self.config_tf.sim_height)
        self.map_extent = (self.config_tf.corner_coords[0], 
                           self.config_tf.corner_coords[0] + self.config_tf.sim_width*self.config_tf.scale2real,
                           self.config_tf.corner_coords[1], 
                           self.config_tf.corner_coords[1] + self.config_tf.sim_height*self.config_tf.scale2real)
        self.human_starts = [self.tf_img2real(x) for x in self._human_starts_img]
        self.robot_starts = [self.tf_img2real(x) for x in self._robot_starts_img]

    def _prepare_global_coordinator(self, graph_fpath: str, map_fpath: str):
        if self.vb:
            print(f'[{self.__class__.__name__}] Preparing global coordinator...')

        self.robot_ids = list(range(len(self.robot_paths)))

        ### Create a schedule dictionary (if external schedule is not available)
        robot_id_column_ = [[rid]*len(self.robot_paths[rid]) for rid in range(len(self.robot_paths))]
        robot_id_column = [item for sublist in robot_id_column_ for item in sublist]
        node_id_column = [node for path in self.robot_paths for node in path]
        schedule_dict = {'robot_id': robot_id_column,
                         'node_id': node_id_column,
                         'ETA': ['none'] * len(node_id_column)}
        
        inflation_margin     = (self.config_robot.vehicle_width/2) / self.config_tf.scale2real
        inflation_margin_mpc = (self.config_robot.vehicle_width/2+self.config_robot.vehicle_margin+0.1) / self.config_tf.scale2real
        self.gpc = GlobalPathCoordinator.from_dict(schedule_dict)
        self.gpc_mpc = GlobalPathCoordinator.from_dict(schedule_dict)
        self.gpc.load_graph_from_json(graph_fpath)
        self.gpc.load_map_from_json(map_fpath, inflation_margin=inflation_margin)
        self.gpc_mpc.load_map_from_json(map_fpath, inflation_margin=inflation_margin_mpc)
        self.gpc.coordinate_convert(self.tf_img2real)
        self.gpc_mpc.coordinate_convert(self.tf_img2real)

        self.static_obstacles = self.gpc.inflated_map.obstacle_coords_list
        self.static_obstacles_mpc = self.gpc_mpc.inflated_map.obstacle_coords_list

    def _prepare_robot_manager(self):
        if self.vb:
            print(f'[{self.__class__.__name__}] Preparing robots...')
        self.robot_manager = RobotManager()

        for rid, rstart in zip(self.robot_ids, self.robot_starts):
            robot = self.robot_manager.create_robot(self.config_robot, motion_model=None, id_=rid)
            robot.set_state(rstart)
            planner = LocalTrajPlanner(self.config_tracker.ts, self.config_tracker.N_hor, self.config_robot.lin_vel_max, verbose=False)
            planner.load_map(self.gpc.inflated_map.boundary_coords, self.gpc.inflated_map.obstacle_coords_list)
            if self.tracker_type == 'mpc':
                controller = TrajectoryTrackerMPC(self.config_tracker, self.config_robot, robot_id=rid, verbose=False)
                controller.load_motion_model(robot.motion_model)
                # controller.set_monitor(monitor_on=False)
            elif self.tracker_type == 'rpp':
                controller = TrajectoryTrackerRPP(self.config_tracker, self.config_robot, robot_id=rid, verbose=False)
            visualizer = CircularObjectVisualizer(self.config_robot.vehicle_width/2, indicate_angle=True)
            self.robot_manager.add_robot(robot, controller, planner, visualizer)
    
    def _prepare_motion_predictor(self, type, config_file_path, ref_image_path, model_suffix='0'):
        if self.vb:
            print(f'[{self.__class__.__name__}] Preparing neural networks...')
        if type is None:
            self.motion_predictor = None
        elif type in ['nll', 'enll', 'bce', 'kld']:
            self.motion_predictor = MotionPredictorEBM(config_file_path, model_suffix, ref_image_path)
        elif type == 'cvm' or type == 'cv':
            self.motion_predictor = MotionPredictorCVM(math.radians(10), ref_image_path)
        elif type == 'sgan':
            self.motion_predictor = MotionPredictorSGAN(config_file_path, ref_image_path)
        else:
            raise ValueError(f'[{self.__class__.__name__}] Invalid motion predictor type: {type}')

    def _prepare_evaluation(self, repeat:int=1):
        self.evaluator = Evaluator(self.robot_ids,
                                   self.config_robot.vehicle_width/2,
                                   self.config_human.human_width,
                                   repeat=repeat)

    def _prepare_pedestrians(self, pedestrian_model:Optional[str]=None):
        """
        Args:
            pedestrian_model: Can be `None` (non-interactive), 'sf' (social force).
        """
        if self.vb:
            print(f'[{self.__class__.__name__}] Preparing agents...')
        self.pedestrian_model = pedestrian_model
        self.humans = [HumanObject(h_s, self.config_robot.ts, self.config_human.human_width, self.config_human.human_stagger) for h_s in self.human_starts]
        human_paths_coords = [self.gpc.current_graph.return_given_path(h_p) for h_p in self.human_paths]
        for i, human in enumerate(self.humans):
            human.set_path(human_paths_coords[i])
            if pedestrian_model == 'sf':
                human.set_social_repulsion(max_distance=5.0, max_angle=math.pi/4, max_force=1.0, opponent_type=RobotObject)
            elif pedestrian_model == 'minisf':
                human.set_social_repulsion(max_distance=3.0, max_angle=math.pi/4, max_force=0.5, opponent_type=RobotObject)
        self.human_visualizers = [CircularObjectVisualizer(self.config_human.human_width, indicate_angle=False) for _ in self.humans]

    @property
    def planner_type(self):
        return self._planner_type

    @property
    def tracker_type(self):
        return self._tracker_type

    @property
    def motion_predictor_type(self):
        return self._motion_predictor_type

    def load(self, 
             planner_type: Optional[str], config_planner_path: str, 
             tracker_type: Optional[str], config_tracker_path: str, 
             config_robot_path: str, config_human_path: str, config_tf_path: str):
        """
        Args:
            planner_type: The type of the planner, 'none' or 'teb'.
            tracker_type: The type of the trajectory tracker, 'mpc' or 'rpp'.
        """
        self._planner_type = planner_type
        self._tracker_type = tracker_type
        self._load_config(config_planner_path, config_tracker_path, config_robot_path, config_human_path)
        self._load_tf(config_tf_path)
        print(f'[{self.__class__.__name__}] Loading complete!')

    def prepare(self, graph_path: str, map_path: str,
                motion_predictor_type: Optional[str], motion_predictor_config_path: str, 
                ref_image_path: str, model_suffix: str,
                pedestrian_model:Optional[str]=None):
        """Prepare the simulation (coordinator, robots, motion predictor, pedestrians)

        Args:
            motion_predictor_type: The type of the motion predictor, 'cvm', 'sgan', 'nll', 'enll', 'bce', 'kld'.
            ref_image_path: The reference image to plot as the background.
            model_suffix: The suffix of the model to load, used to quickly load different models.
            pedestrian_model: The model for the pedestrian, can be `None` (non-interactive), 'sf' (social force).
        """
        self._motion_predictor_type = motion_predictor_type

        prt_process = 'Preparing: Coordinator'
        process_timer = timeit.default_timer()
        print(f'[{self.__class__.__name__}] {prt_process}', end=' ')
        self._prepare_global_coordinator(graph_path, map_path)

        prt_process += f' {round(timeit.default_timer()-process_timer, 3)}s | Manager'
        process_timer = timeit.default_timer()
        print(f'\r[{self.__class__.__name__}] {prt_process}', end=' ')
        self._prepare_robot_manager()
        self.generic_planner = None
        if self.planner_type == 'teb':
            self.generic_planner = TrajectoryPlannerTEB(self.config_planner, self.config_robot, safe_factor=3.0, safe_margin=0.1, verbose=self.vb)

        prt_process += f' {round(timeit.default_timer()-process_timer, 3)}s | Predictor'
        process_timer = timeit.default_timer()
        print(f'\r[{self.__class__.__name__}] {prt_process}', end=' ')
        self._prepare_motion_predictor(motion_predictor_type, motion_predictor_config_path, ref_image_path, model_suffix)

        self.gpc.load_img_map(ref_image_path)
        for rid, rstart in zip(self.robot_ids, self.robot_starts):
            path_coords, path_times = self.gpc.get_robot_schedule(rid)
            self.robot_manager.add_schedule(rid, rstart, path_coords, path_times)

        prt_process += f' {round(timeit.default_timer()-process_timer, 3)}s | Pedestrian'
        process_timer = timeit.default_timer()
        print(f'\r[{self.__class__.__name__}] {prt_process}', end=' ')
        self._prepare_pedestrians(pedestrian_model)

        prt_process += f' {round(timeit.default_timer()-process_timer, 3)}s | Visualizer'
        process_timer = timeit.default_timer()
        print(f'\r[{self.__class__.__name__}] {prt_process}', end=' ')
        if self.viz:
            self.main_plotter = MpcPlotInLoop(self.config_robot)
            # graph_manager = self.gpc.current_graph
            graph_manager = None
            self.main_plotter.plot_in_loop_pre(self.gpc.img_map, self.gpc.inflated_map, graph_manager, map_extend=self.map_extent)
            
            for rid in self.robot_ids:
                robot = self.robot_manager.get_robot(rid)
                planner = self.robot_manager.get_planner(rid)
                controller = self.robot_manager.get_controller(rid)
                visualizer = self.robot_manager.get_visualizer(rid)
                self.main_plotter.add_object_to_pre(rid,
                                                    planner.ref_traj,
                                                    controller.state,
                                                    controller.final_goal,
                                                    color=self.color_list[rid])
                visualizer.plot(self.main_plotter.map_ax, *robot.state, object_color=self.color_list[rid])

            for _, (human, human_vis) in enumerate(zip(self.humans, self.human_visualizers)):
                human_vis.plot(self.main_plotter.map_ax, x=human.state[0], y=human.state[1], object_color='m')

        prt_process += f' {round(timeit.default_timer()-process_timer, 3)}s'
        print(f'\r[{self.__class__.__name__}] {prt_process}')

        print(f'[{self.__class__.__name__}] Preparation complete!')

    def reset(self):
        if self.viz:
            self.main_plotter.close()
            plt.close('all')

        self._prepare_robot_manager()
        for rid, rstart in zip(self.robot_ids, self.robot_starts):
            path_coords, path_times = self.gpc.get_robot_schedule(rid)
            self.robot_manager.add_schedule(rid, rstart, path_coords, path_times)
        self._prepare_pedestrians(pedestrian_model=self.pedestrian_model)

        if self.viz:
            self.main_plotter = MpcPlotInLoop(self.config_robot)
            # graph_manager = self.gpc.current_graph
            graph_manager = None
            self.main_plotter.plot_in_loop_pre(self.gpc.img_map, self.gpc.inflated_map, graph_manager, map_extend=self.map_extent)
            
            for rid in self.robot_ids:
                robot = self.robot_manager.get_robot(rid)
                planner = self.robot_manager.get_planner(rid)
                controller = self.robot_manager.get_controller(rid)
                visualizer = self.robot_manager.get_visualizer(rid)
                self.main_plotter.add_object_to_pre(rid,
                                                    planner.ref_traj,
                                                    controller.state,
                                                    controller.final_goal,
                                                    color=self.color_list[rid])
                visualizer.plot(self.main_plotter.map_ax, *robot.state, object_color=self.color_list[rid])

            for i, (human, human_vis) in enumerate(zip(self.humans, self.human_visualizers)):
                human_vis.plot(self.main_plotter.map_ax, *human.state, object_color='r')

        print(f'[{self.__class__.__name__}] Reset the simulation!')


    def run_motion_prediction(self, eps=0.5, min_sample=10, enlarge=3.0, extra_margin=0.0, debug=False):
        """Run motion prediction for all agents and combine the results to get the nominal obstacles.

        Args:
            debug: If True, also return the outputs from the network. Defaults to False.

        Returns:
            clusters_list: A list of clusters, each cluster is a list of points.
            mu_list_list: A list of means of the clusters.
            std_list_list: A list of standard deviations of the clusters.
            conf_list_list: A list of confidence of the clusters.
            logits_for_all (if debug): The logits prediction, [CxHxW].
            prob_maps_for_all (if debug): The probability map, [CxHxW]
        """
        if self.motion_predictor_type not in ['nll', 'enll', 'bce', 'kld']:
            debug = False # Only EBM supports debug mode

        ### Current positions
        curr_mu_list = [human.state[:2].tolist() for human in self.humans]
        curr_std_list = [[self.config_human.human_width, self.config_human.human_width] for _ in self.humans]
        
        ### Run prediction
        if self.motion_predictor_type in ['nll', 'enll', 'bce', 'kld']:
            assert isinstance(self.motion_predictor, MotionPredictorEBM)
            past_traj_NN = [self.tf_img2real(x, False) for x in self.humans[0].past_traj]
            pred_samples_all = self.motion_predictor.get_motion_prediction_samples(past_traj_NN, rescale=self.config_tf.scale2nn)
            if debug:
                _, _, _, _, logits, prob_maps = self.motion_predictor.get_motion_prediction(past_traj_NN, rescale=self.config_tf.scale2nn, debug=True)
                logits_for_all = [logits]
                prob_maps_for_all = [prob_maps]

            for human in self.humans[1:]:
                past_traj_NN = [self.tf_img2real(x, False) for x in human.past_traj] # convert the real world coordinates to the image coordinates
                if debug:
                    _, _, _, _, logits, prob_maps = self.motion_predictor.get_motion_prediction(past_traj_NN, rescale=self.config_tf.scale2nn, debug=True)
                    logits_for_all.append(logits)
                    prob_maps_for_all.append(prob_maps)
                pred_samples = self.motion_predictor.get_motion_prediction_samples(past_traj_NN, rescale=self.config_tf.scale2nn)
                pred_samples_all = [np.concatenate((x,y), axis=0) for x,y in zip(pred_samples_all, pred_samples)]
            pred_samples_all = [self.tf_img2real.cvt_coords(x[:,0], x[:,1]) for x in pred_samples_all] # each element is [total_num_samples, 2]

        elif self.motion_predictor_type == 'cvm':
            assert isinstance(self.motion_predictor, MotionPredictorCVM)
            pred_samples_all = self.motion_predictor.get_motion_prediction_samples(self.humans[0].past_traj, num_samples=20, pred_len=self.config_tracker.N_hor)
            for human in self.humans[1:]:
                pred_samples = self.motion_predictor.get_motion_prediction_samples(human.past_traj, num_samples=20, pred_len=self.config_tracker.N_hor)
                pred_samples_all = [np.concatenate((x,y), axis=0) for x,y in zip(pred_samples_all, pred_samples)]

        elif self.motion_predictor_type == 'sgan':
            assert isinstance(self.motion_predictor, MotionPredictorSGAN)
            ### Solution 1: S-GAN with only humans
            # all_past_traj_NN = [human.past_traj for human in self.humans]
            ### Solution 2: S-GAN with humans and robots
            all_past_traj = [human.past_traj for human in self.humans] + [robot.past_traj for robot in self.robot_manager.get_all_robots()]

            pred_samples_all_list = self.motion_predictor.get_motion_prediction_samples(
                all_past_traj, rescale=self.config_tf.scale2nn, num_samples=20, 
                sampling_time=self.config_robot.ts, pred_len=self.config_tracker.N_hor)
            pred_samples_all_list = pred_samples_all_list[:len(self.humans)] # get rid of the robot samples if any
            pred_samples_all = [np.concatenate([x[i] for x in pred_samples_all_list], axis=0) for i in range(len(pred_samples_all_list[0]))]
        
        ### CGF
        clusters_list, mu_list_list, std_list_list, conf_list_list = self.motion_predictor.clustering_and_fitting_from_samples(np.array(pred_samples_all), eps=eps, min_sample=min_sample, enlarge=enlarge, extra_margin=extra_margin)
        mu_list_list = [curr_mu_list] + mu_list_list
        std_list_list = [curr_std_list] + std_list_list
        conf_list_list = [[1.0]*len(curr_mu_list)] + conf_list_list
        if debug:
            return clusters_list, mu_list_list, std_list_list, conf_list_list, logits_for_all, prob_maps_for_all
        else:
            return clusters_list, mu_list_list, std_list_list, conf_list_list
    
    def run_motion_prediction_for_individuals(self, eps=0.5, min_sample=10, enlarge=3.0, extra_margin=0.0, debug=False):
        """Samples from the same agent were clustered by DBSCAN at each time step (collaborate with Junjie Hu).

        Args:
            debug: If True, also return the outputs from the network. Defaults to False.

        Returns:
            clusters_list: A list of clusters, each cluster is a list of points.
            mu_list_list: A list of means of the clusters.
            std_list_list: A list of standard deviations of the clusters.
            conf_list_list: A list of confidence of the clusters.
            logits_for_all (if debug): The logits prediction, [CxHxW].
            prob_maps_for_all (if debug): The probability map, [CxHxW]
        """
        ### Current positions
        curr_mu_list = [human.state[:2].tolist() for human in self.humans]
        curr_std_list = [[self.config_human.human_width, self.config_human.human_width] for _ in self.humans]
        
        ### Run prediction
        if self.motion_predictor_type == 'enll':
            assert isinstance(self.motion_predictor, MotionPredictorEBM)
            past_traj_NN = [self.tf_img2real(x, False) for x in self.humans[0].past_traj]
            pred_samples_one = self.motion_predictor.get_motion_prediction_samples(past_traj_NN, rescale=self.config_tf.scale2nn)
            pred_samples_one = [self.tf_img2real.cvt_coords(x[:,0], x[:,1]) for x in pred_samples_one]
            if debug:
                _, _, _, _, logits, prob_maps = self.motion_predictor.get_motion_prediction(past_traj_NN, rescale=self.config_tf.scale2nn, debug=True)
                logits_for_all = [logits]
                prob_maps_for_all = [prob_maps]

            clusters_list, mu_list_list, std_list_list, conf_list_list = self.motion_predictor.clustering_and_fitting_from_samples(np.array(pred_samples_one), eps=eps, min_sample=min_sample, enlarge=enlarge, extra_margin=extra_margin)
            
            for human in self.humans[1:]:
                past_traj_NN = [self.tf_img2real(x, False) for x in human.past_traj] # convert the real world coordinates to the image coordinates
                if debug:
                    _, _, _, _, logits, prob_maps = self.motion_predictor.get_motion_prediction(past_traj_NN, rescale=self.config_tf.scale2nn, debug=True)
                    logits_for_all.append(logits)
                    prob_maps_for_all.append(prob_maps)
                pred_samples = self.motion_predictor.get_motion_prediction_samples(past_traj_NN, rescale=self.config_tf.scale2nn)
                pred_samples = [self.tf_img2real.cvt_coords(x[:,0], x[:,1]) for x in pred_samples]
                curr_clusters_list, curr_mu_list_list, curr_std_list_list, curr_conf_list_list = self.motion_predictor.clustering_and_fitting_from_samples(np.array(pred_samples), eps=eps, min_sample=min_sample, enlarge=enlarge, extra_margin=extra_margin)

                for t in range(len(clusters_list)):
                    clusters_list[t].extend(curr_clusters_list[t])
                mu_list_list = [m1 + m2 for m1, m2 in zip(mu_list_list, curr_mu_list_list)]
                std_list_list = [m1 + m2 for m1, m2 in zip(std_list_list, curr_std_list_list)]
                conf_list_list = [m1 + m2 for m1, m2 in zip(conf_list_list, curr_conf_list_list)]

            mu_list_list = [curr_mu_list] + mu_list_list
            std_list_list = [curr_std_list] + std_list_list
            conf_list_list = [[1.0]*len(curr_mu_list)] + conf_list_list

        elif self.motion_predictor_type == 'cvm':
            assert isinstance(self.motion_predictor, MotionPredictorCVM)
            logits_for_all = None
            prob_maps_for_all = None
            pred_samples_all = [self.motion_predictor.get_motion_prediction_samples(self.humans[0].past_traj, num_samples=20, pred_len=self.config_tracker.N_hor)]
            for human in self.humans[1:]:
                pred_samples = self.motion_predictor.get_motion_prediction_samples(human.past_traj, num_samples=20, pred_len=self.config_tracker.N_hor)
                pred_samples_all.append(pred_samples)
            clusters_list, mu_list_list, std_list_list, conf_list_list = self.motion_predictor.clustering_and_fitting_from_samples(np.array(pred_samples_all[0]), eps=eps, min_sample=min_sample, enlarge=enlarge, extra_margin=extra_margin)
            for pred_samples in pred_samples_all[1:]:
                curr_clusters_list, curr_mu_list_list, curr_std_list_list, curr_conf_list_list = self.motion_predictor.clustering_and_fitting_from_samples(np.array(pred_samples), eps=eps, min_sample=min_sample, enlarge=enlarge, extra_margin=extra_margin)
                
                for t in range(len(clusters_list)):
                    clusters_list[t].extend(curr_clusters_list[t])
                mu_list_list = [m1 + m2 for m1, m2 in zip(mu_list_list, curr_mu_list_list)]
                std_list_list = [m1 + m2 for m1, m2 in zip(std_list_list, curr_std_list_list)]
                conf_list_list = [m1 + m2 for m1, m2 in zip(conf_list_list, curr_conf_list_list)]

            mu_list_list = [curr_mu_list] + mu_list_list
            std_list_list = [curr_std_list] + std_list_list
            conf_list_list = [[1.0]*len(curr_mu_list)] + conf_list_list
            
        elif self.motion_predictor_type == 'sgan':
            assert isinstance(self.motion_predictor, MotionPredictorSGAN)
            all_past_traj = [human.past_traj for human in self.humans] + [robot.past_traj for robot in self.robot_manager.get_all_robots()]
            pred_samples_all = self.motion_predictor.get_motion_prediction_samples(
                all_past_traj, rescale=self.config_tf.scale2nn, num_samples=20, 
                sampling_time=self.config_robot.ts, pred_len=self.config_tracker.N_hor)
            pred_samples_all = pred_samples_all[:len(self.humans)] # get rid of the robot samples if any
            clusters_list, mu_list_list, std_list_list, conf_list_list = self.motion_predictor.clustering_and_fitting_from_samples(np.array(pred_samples_all[0]), eps=eps, min_sample=min_sample, enlarge=enlarge, extra_margin=extra_margin)
            for pred_samples in pred_samples_all[1:]:
                curr_clusters_list, curr_mu_list_list, curr_std_list_list, curr_conf_list_list = self.motion_predictor.clustering_and_fitting_from_samples(np.array(pred_samples), eps=eps, min_sample=min_sample, enlarge=enlarge, extra_margin=extra_margin)
                
                for t in range(len(clusters_list)):
                    clusters_list[t].extend(curr_clusters_list[t])
                mu_list_list = [m1 + m2 for m1, m2 in zip(mu_list_list, curr_mu_list_list)]
                std_list_list = [m1 + m2 for m1, m2 in zip(std_list_list, curr_std_list_list)]
                conf_list_list = [m1 + m2 for m1, m2 in zip(conf_list_list, curr_conf_list_list)]

            mu_list_list = [curr_mu_list] + mu_list_list
            std_list_list = [curr_std_list] + std_list_list
            conf_list_list = [[1.0]*len(curr_mu_list)] + conf_list_list
        else:
            raise ValueError(f'[{self.__class__.__name__}] Predictor type: {self.motion_predictor_type} is not supported for individual clustering.')
        
        if debug:
            return clusters_list, mu_list_list, std_list_list, conf_list_list, logits_for_all, prob_maps_for_all
        else:
            return clusters_list, mu_list_list, std_list_list, conf_list_list

    def run_one_step(self, current_time: float, num_repeat: int, extra_debug_panel:Optional[list[Axes]]=None, auto_run:bool=True):
        """Run one step of the simulation.

        Args:
            current_time: The current time (real time but not time step).
            num_repeat: The index of the current repeat.
            extra_debug_panel: _description_. Defaults to None.

        Returns:
            clusters_list: A list (n=horizon) of clusters, each cluster is a list of points.
            mu_list_list: A list of means of the clusters.
            std_list_list: A list of standard deviations of the clusters.
            conf_list_list: A list of confidence of the clusters.
        """
        kt = current_time / self.config_tracker.ts

        ### Run motion prediction
        start_mmp_time = timeit.default_timer()
        if self.motion_predictor_type is not None:
            if extra_debug_panel is not None:
                look_at_step = 20
                clusters_list, mu_list_list, std_list_list, conf_list_list, logits_for_all, prob_maps_for_all = self.run_motion_prediction(extra_margin=self.config_human.human_width, debug=True)
                [ax.cla() for ax in extra_debug_panel] # type: ignore
                extra_debug_panel[0].imshow(logits_for_all[0][look_at_step-1]+self.gpc.img_map()/255.0*logits_for_all[0][look_at_step-1].max().numpy(), cmap='gray')
                extra_debug_panel[1].imshow(prob_maps_for_all[0][look_at_step-1], cmap='gray', extent=self.map_extent)
                for cluster in clusters_list[look_at_step-1]:
                    extra_debug_panel[1].scatter(cluster[:,0], cluster[:,1])
            else:
                clusters_list, mu_list_list, std_list_list, conf_list_list = self.run_motion_prediction(extra_margin=self.config_human.human_width)
                # clusters_list, mu_list_list, std_list_list, conf_list_list = self.run_motion_prediction_for_individuals(extra_margin=self.config_human.human_width)
        else:
            clusters_list = [[human.state[:2].tolist() for human in self.humans]]
            mu_list_list = [[human.state[:2].tolist() for human in self.humans]]
            std_list_list = [[[self.config_human.human_width, self.config_human.human_width] for _ in self.humans]]
            conf_list_list = [[1.0]*len(self.humans)]
        solve_mmp_time = timeit.default_timer() - start_mmp_time
        self.evaluator.append_mmp_solve_time(solve_mmp_time)

        # if self.tracker_type in ['mpc']:
        n_obs = max([len(x) for x in mu_list_list])
        n_obs = min(n_obs, 15)
        dyn_obs_list = [[[0, 0, 0, 0, 0, 1]]*(self.config_tracker.N_hor+1) for _ in range(n_obs)]
        if self.planner_type not in ['teb']:
            for Tt, (mu_list, std_list, conf_list) in enumerate(zip(mu_list_list, std_list_list, conf_list_list)):
                for Nn, (mu, std, conf) in enumerate(zip(mu_list, std_list, conf_list)): # at each time offset
                    if Nn > 14:
                        break
                    dyn_obs_list[Nn][Tt] = [mu[0], mu[1], std[0], std[1], 0, conf] # for each obstacle

        ### Human step
        sf_viz_list = [] # type: ignore
        for i, human in enumerate(self.humans):
            agents = [self.robot_manager.get_robot(rid).robot_object for rid in self.robot_ids]
            social_force, _, attenuation_factor = human.get_social_repulsion(agents) # will work only if the human is set to social force model
            action = human.run_step(self.config_human.human_vel_max, social_force=social_force, attenuation_factor=attenuation_factor)

            # aaxx = self.main_plotter.map_ax # XXX
            # if action is not None:
            #     # print(f'Human {i} attenuation_factor: {attenuation_factor}')
            #     sf_viz1 = aaxx.quiver(human.state[0], human.state[1], action[0]-social_force[0], action[1]-social_force[1], angles='xy', scale_units='xy', scale=0.5, color='gray')
            #     sf_viz2 = aaxx.quiver(human.state[0], human.state[1], action[0], action[1], angles='xy', scale_units='xy', scale=0.5, color='k')
            #     sf_viz_list.extend([sf_viz1, sf_viz2])
            #     human.plot_social_force(aaxx, color='b', plot_all=True, length_inverse_scale=0.5)

            if self.viz:
                self.human_visualizers[i].update(*human.state)
        
        ### Run Tracker
        dynamic_obstacles = [human.state[:2].tolist() for human in self.humans] # for collision-checking
        dynamic_obstacles_with_predictions = [mu for mu_list in mu_list_list for mu in mu_list]

        for rid in self.robot_ids:
            if self.evaluator.eval_results[rid]["complete_results"][num_repeat]:
                continue # skip if this robot has completed

            robot = self.robot_manager.get_robot(rid)
            planner = self.robot_manager.get_planner(rid)
            controller = self.robot_manager.get_controller(rid)
            visualizer = self.robot_manager.get_visualizer(rid)

            # ### Threat assessment
            # for human in self.humans:
            #     ego_velocity = (robot.state[:2] - robot.past_traj[-1][:2]) / self.config_robot.ts
            #     target_velocity = (human.state[:2] - human.past_traj[-1][:2]) / self.config_robot.ts
            #     threat_assessment(ego_position=robot.state[:2], ego_velocity=ego_velocity,
            #                       target_position=human.state[:2], target_velocity=target_velocity,
            #                       safe_reaction_time=0.5, safe_distance=1)


            controller.set_current_state(robot.state)
            if controller.finishing:
                ref_states, ref_speed, *_ = planner.get_local_ref(kt*self.config_tracker.ts, (float(robot.state[0]), float(robot.state[1])))
            else:
                ref_states, ref_speed, *_ = planner.get_local_ref(kt*self.config_tracker.ts, (float(robot.state[0]), float(robot.state[1])), external_ref_speed=controller.base_speed)
            if self.planner_type is not None:
                assert self.generic_planner is not None
                self.generic_planner.set_ref_states(robot.state, ref_states, ref_speed)
                ref_states, _ = self.generic_planner.run_step(obstacles=dynamic_obstacles,
                                                              obstacle_radius=self.config_human.human_width*1.5)
            
            ### Optimize the trajectory
            # if self.tracker_type in ['mpc']:
            #     if self.robot_manager.get_pred_states(rid) is not None:
            #         last_pred_states = self.robot_manager.get_pred_states(rid)
            #         if np.dot(ref_states[-1, :2]-ref_states[0, :2], last_pred_states[-1, :2]-last_pred_states[0, :2]) < 0:
            #             ref_states = ref_states
            #         else:
            #             ref_states = ref_states*0.9 + last_pred_states*0.1
            ###

            controller.set_ref_states(ref_states, ref_speed=ref_speed)

            if (len(self.robot_manager) > 1) and self.tracker_type in ['mpc']:
                other_robot_states = self.robot_manager.get_other_robot_states(rid, config_mpc=self.config_tracker)
            else:
                other_robot_states = None

            start_tracker_time = timeit.default_timer()
            if self.tracker_type == 'mpc':
                assert isinstance(controller, TrajectoryTrackerMPC)
                actions, pred_states, current_refs, debug_info = controller.run_step(static_obstacles=self.static_obstacles_mpc,
                                                                                     full_dyn_obstacle_list=dyn_obs_list,
                                                                                     other_robot_states=other_robot_states,
                                                                                     map_updated=True)
                action = actions[-1]
            elif self.tracker_type == 'rpp':
                assert isinstance(controller, TrajectoryTrackerRPP)
                actions, pred_states, current_refs, debug_info = controller.run_step(obstacles=dynamic_obstacles,
                                                                                     obstacle_radius=self.config_human.human_width*1.0)
                action = actions[0]
            solve_tracker_time = timeit.default_timer() - start_tracker_time
            self.evaluator.append_tracker_solve_time(rid, solve_tracker_time)

            self.robot_manager.set_pred_states(rid, np.asarray(pred_states))

            assert isinstance(action, np.ndarray)

            ### Regulate the action
            # regulated_v = action[0] * max(0.8, (self.config_robot.ang_vel_max-abs(action[1])) / self.config_robot.ang_vel_max) # type: ignore
            # action[0] = regulated_v

            ### Robot step
            if self.tracker_type == 'mpc':
                if (action[0]<self.config_robot.lin_vel_min*0.9 or action[0]==1.5 or debug_info['cost']>1e3):
                    controller.restart_solver()
                    # action[0] = -0.2
                # elif (debug_info['cost'] < 20 and action[0] < 0.1):
                #     controller.restart_solver()
                #     action[0] = 0.2
            robot.step(action)

            if controller.check_termination_condition(external_check=planner.idle):
                self.evaluator.check_completion(rid, num_repeat, True)

            ### Check collisions
            other_states = [r.state for r in self.robot_manager.get_all_robots() if r.id_ != rid]
            have_collision = self.evaluator.check_collision(rid, num_repeat, robot.state, other_states, self.static_obstacles, dynamic_obstacles)
            # if have_collision:
            #     print(f"COLLISION COST: {debug_info['cost']}")
            #     input("Detect collision! Press Enter to continue...")
            if self.eval:
                self.evaluator.calc_minimal_dynamic_obstacle_distance(rid, num_repeat, robot.state, dynamic_obstacles)

            if self.viz:
                self.main_plotter.update_plot(rid, kt, action, robot.state, debug_info['cost'], np.asarray(pred_states), current_refs)
                visualizer.update(*robot.state)

            if self.vb:
                assert action is not None, f"Action is None for robot {rid}."
                prt_action = f'Actions:({round(action[0], 4)}, {round(action[1], 4)});'
                prt_state  = f'Robot state: R/T {[round(x,4) for x in robot.state]}/{[round(x,4) for x in controller.state]};'
                prt_cost   = f"Cost:{round(debug_info['cost'],4)}."
                print(f"[{self.__class__.__name__}] Time:{current_time:.2f} | Robot {rid} |", prt_action, prt_state, prt_cost)
            else:
                print(f"[{self.__class__.__name__}] Time:{current_time:.2f}", end='\r')

        if self.viz:
            zoom_in = [-6, 10, -1, 14] # for long-term
            self.main_plotter.plot_in_loop(
                dyn_obstacle_list=dyn_obs_list, 
                time=current_time, 
                autorun=auto_run, 
                # zoom_in=zoom_in, 
                # save_path=f'Demo/{int(current_time/self.config_tracker.ts)}.png',
                temp_plots=sf_viz_list,
                # temp_objects=debug_info['closest_obstacle_list']
            )
            for i, human in enumerate(self.humans):
                self.human_visualizers[i].update(*human.state)

        return clusters_list, mu_list_list, std_list_list, conf_list_list

    def run_once(self, repeat:int=1, time_step_out:Optional[float]=None, extra_debug_panel:Optional[list[Axes]]=None, auto_run:bool=True):
        """Run the simulation once for a given number of repeats.

        Args:
            repeat: The number of repeats. Defaults to 1.
            time_step_out: The time to stop the simulation if not None. Defaults to None.
            extra_debug_panel: List of Axes for extra debug panel if not None. Defaults to None.
            auto_run: If True, the plot will be updated automatically (without plt.waitforbuttonpress). Defaults to True.
        """
        self._prepare_evaluation(repeat=repeat)
        if repeat > 1 and self.viz:
            warnings.warn("Try to avoid visualization when repeat > 1.")

        first_run = True
        for i in range(repeat):
            print(f'[{self.__class__.__name__}] Repeat {i+1}/{repeat}:')

            if first_run:
                assert i==0, "The first run should be the first repeat (index 0)."
                time.sleep(3.0) # wait for the plot to be ready
                first_run = False

            current_time = 0.0
            total_complete = False
            any_collision = False
            while (not total_complete) and (not any_collision):
                self.run_one_step(current_time=current_time,
                                  num_repeat=i, 
                                  extra_debug_panel=extra_debug_panel,
                                  auto_run=auto_run)
                total_complete = all([self.evaluator.eval_results[rid]["complete_results"][i] for rid in self.robot_ids])
                any_collision = any([self.evaluator.eval_results[rid]["collision_results"][i] for rid in self.robot_ids])
                
                if (time_step_out is not None) and (current_time+1e-6 >= time_step_out):
                    break
                current_time += self.config_tracker.ts
                if self.viz:
                    time.sleep(0.5)
                else:
                    time.sleep(0.01)

            if total_complete and self.eval:
                for rid in self.robot_ids:
                    self.evaluator.calc_action_smoothness(rid, i, self.robot_manager.get_controller(rid).past_actions)
                    self.evaluator.calc_minimal_obstacle_distance(rid, i, self.robot_manager.get_controller(rid).past_states, self.static_obstacles)
                    self.evaluator.calc_deviation_distance(rid, i, ref_traj=self.robot_manager.get_planner(rid).ref_traj, actual_traj=self.robot_manager.get_controller(rid).past_states)

            if not self.vb:
                print() # print a new line

            print(f'[{self.__class__.__name__}] Repeat {i+1} finished. Any collision: {any_collision}. All complete: {total_complete}. Timeout: {(not total_complete) and (not any_collision)}.')

            if self.viz:
                if any_collision:
                    input(f"[{self.__class__.__name__}] Collision detected. Press Enter to continue...")
                # self.main_plotter.show()

            if i < repeat-1:

                self.reset()

        if self.viz: 
            if auto_run:
                input("Press Enter to continue...")
            self.main_plotter.close()
            plt.close('all')


    def report(self, save_dir:Optional[str]=None):
        for rid in self.robot_ids:
            save_path = None     
            if save_dir is not None:
                save_path = f'{save_dir}/robot_{rid}_results.json'
            self.evaluator.report(rid, full_report=self.eval, save_path=save_path)


if __name__ == '__main__':
    import os

    predictor_type = 'kld' # None, 'nll/enll/kld/bce', 'cvm', or 'sgan'
    tracker_type = 'mpc' # 'mpc', 'rpp'
    planner_type = None # None, 'teb'
    scenario_index = 1 # 1-4, 20
    pedestrian_model = 'None' # None, 'sf', 'minisf'
    auto_run = True 
    evaluation = False
    repeat = 1
    time_step = 30.0 # seconds. 200 for long-term, 30 for short-term

    project_dir = pathlib.Path(__file__).resolve().parents[1]

    ### Check planner type
    if planner_type is None:
        cfg_planner_path = 'none'
    elif planner_type == 'teb':
        cfg_planner_path = os.path.join(project_dir, 'config', 'teb.yaml')
    else:
        raise ValueError(f'Invalid planner type: {planner_type}')

    ### Check tracker type
    if tracker_type == 'mpc':
        cfg_tracker_path = os.path.join(project_dir, 'config', 'mpc_fast.yaml')
    elif tracker_type == 'rpp':
        cfg_tracker_path = os.path.join(project_dir, 'config', 'rpp.yaml')
    else:
        raise ValueError(f'Invalid tracker type: {tracker_type}')
    
    ### Check predictor type
    if predictor_type is None:
        motion_predictor_config_path = 'none'
    elif predictor_type == 'enll':
        motion_predictor_config_path = os.path.join(project_dir, 'config', 'wsd_1t20_poselu_enll_train.yaml')
    elif predictor_type in ['bce', 'kld', 'nll']:
        motion_predictor_config_path = os.path.join(project_dir, 'config', 'wsd_1t20_none_'+predictor_type+'_train.yaml')
    elif predictor_type == 'cvm':
        motion_predictor_config_path = 'none' # no need for config file
    elif predictor_type == 'sgan':
        motion_predictor_config_path = os.path.join(project_dir, 'model', 'sgan_eth12')
    else:
        raise ValueError(f'Invalid predictor type: {predictor_type}')

    ### Other paths
    cfg_robot_path = os.path.join(project_dir, 'config', 'spec_robot.yaml')
    cfg_human_path = os.path.join(project_dir, 'config', 'spec_human.yaml')
    cfg_tf_path = os.path.join(project_dir, 'config', 'global_setting_warehouse.yaml')

    graph_path = os.path.join(project_dir, 'data', 'warehouse_sim_original', 'graph.json')
    map_path = os.path.join(project_dir, 'data', 'warehouse_sim_original', 'map.json')
    ref_image_path = os.path.join(project_dir, 'data', 'warehouse_sim_original', 'background.png')

    mb = MainBase(scenario_index=scenario_index, evaluation=evaluation, visualization=(not evaluation), verbose=False)
    mb.load(planner_type, cfg_planner_path, tracker_type, cfg_tracker_path, cfg_robot_path, cfg_human_path, cfg_tf_path)
    mb.prepare(graph_path, map_path, predictor_type, motion_predictor_config_path, ref_image_path=ref_image_path, model_suffix='0',
               pedestrian_model=pedestrian_model)

    # fig_debug, axes_debug = plt.subplots(1, 2) 

    mb.run_once(repeat=repeat, time_step_out=time_step, extra_debug_panel=None, auto_run=auto_run)
    if evaluation:
        mb.report(save_dir='./')

    print(f'Finish with predictor: {predictor_type}, tracker: {tracker_type}, planner: {planner_type}, scenario: {scenario_index}, human model: {pedestrian_model}.')


    




