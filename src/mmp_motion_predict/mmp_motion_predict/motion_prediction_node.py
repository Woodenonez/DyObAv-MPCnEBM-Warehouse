import os
import time

import numpy as np

import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory # type: ignore

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from mmp_interfaces.msg import HumanTrajectory, HumanTrajectoryArray # type: ignore
from mmp_interfaces.msg import MotionPredictionResult # type: ignore

from .motion_prediction import MotionPredictor
from .map_tf import ScaleOffsetReverseTransform


HUMAN_SIZE = 0.2 # meter


class MotionPredictionNode(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.get_logger().info(f"{self.__class__.__name__} init..")
        self.get_logger().info(f"Waiting for prediction model to load...")

        time.sleep(1) # wait for other nodes to start

        pkg_root_dir = get_package_share_directory('mmp_motion_predict')

        self.declare_parameter('timer_period', 0.2)
        self.timer_period = self.get_parameter('timer_period').value

        # self.declare_parameter('config_file_name', 'wsd_1t20_poselu_enll_train.yaml')
        self.declare_parameter('config_file_name', 'zpd_1t20_poselu_enll_train.yaml')
        self.config_file_name = self.get_parameter('config_file_name').value
        self.config_file_path = os.path.join(pkg_root_dir, 'config', self.config_file_name)

        # self.ref_img_path = os.path.join(pkg_root_dir, 'data', 'warehouse_sim_original', 'background.png') # NOTE
        self.ref_img_path = os.path.join(pkg_root_dir, 'data', 'zospital_sim_original', 'background.png') # NOTE

        # Timer for publishing cmd_vel
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Subscriber to human trajectory
        human_traj_array_name   = 'human_traj_array'
        self.traj_subscription = self.create_subscription(
            HumanTrajectoryArray,
            human_traj_array_name,
            self.traj_callback,
            10
        )

        # Publisher for cmd_vel
        motion_prediction_result_name = 'motion_prediction_result'
        self.motion_prediction_publisher = self.create_publisher(
            MotionPredictionResult,
            motion_prediction_result_name,
            10
        )
        self._init_motion_prediction_result_msg()

        # Publisher for the schedule/path visualization
        viz_msg_name = f'motion_prediction_viz'
        self.motion_prediction_viz_publisher = self.create_publisher(
            MarkerArray,
            viz_msg_name,
            10
        )
        self.last_num_markers = 0

        self.motion_predictor = MotionPredictor(config_file_path=self.config_file_path, model_suffix='1', ref_image_path=self.ref_img_path)
        # self.tf_img2real = ScaleOffsetReverseTransform(
        #     scale=0.1, 
        #     offsetx_after=-15.0, 
        #     offsety_after=-15.0, 
        #     y_reverse=(not False), 
        #     y_max_before=293) # NOTE global_setting_warehouse.yaml
        self.tf_img2real = ScaleOffsetReverseTransform(
            scale=0.05, 
            offsetx_after=-18.0, 
            offsety_after=-8.0, 
            y_reverse=(not False), 
            y_max_before=321) # NOTE global_setting_zospital.yaml
        self.traj_received = False

        self.get_logger().info(f"{self.__class__.__name__} init done.")

    def _init_motion_prediction_result_msg(self):
        self.motion_prediction_msg = MotionPredictionResult()
        self.motion_prediction_msg.header.stamp = self.get_clock().now().to_msg()
        self.motion_prediction_msg.header.frame_id = 'map'

    def timer_callback(self):
        if not self.traj_received:
            self.get_logger().debug("Waiting for human trajectory message...")
            return
        if len(self.trajs) == 0:
            self.get_logger().debug("No human trajectory received.")
            return
        
        ### Current positions
        curr_mu_list = [traj[-1] for traj in self.trajs] # the last point of each trajectory is the current position
        curr_std_list = [[HUMAN_SIZE, HUMAN_SIZE] for _ in self.trajs]
        
        trajs_nn = [[tuple(self.tf_img2real(x, False)) for x in traj] for traj in self.trajs]
        
        if len(trajs_nn[0]) == 1:
            past_traj_NN = trajs_nn[0]
        else:
            past_traj_NN = trajs_nn[0][:-1]
        pred_samples_all = self.motion_predictor.get_motion_prediction_samples(past_traj_NN, rescale=1.0)
        for traj_nn in trajs_nn[1:]:
            if len(trajs_nn[0]) == 1:
                past_traj_NN = traj_nn
            else:
                past_traj_NN = traj_nn[:-1]
            pred_samples = self.motion_predictor.get_motion_prediction_samples(past_traj_NN, rescale=1.0)
            pred_samples_all = [np.concatenate((x,y), axis=0) for x,y in zip(pred_samples_all, pred_samples)]
        pred_samples_all = [self.tf_img2real.cvt_coords(x[:,0], x[:,1]) for x in pred_samples_all]
        ### CGF
        clusters_list, mu_list_list, std_list_list, conf_list_list = self.motion_predictor.clustering_and_fitting_from_samples(np.array(pred_samples_all), eps=0.5, min_sample=10, enlarge=3.0, extra_margin=HUMAN_SIZE)
        mu_list_list = [curr_mu_list] + mu_list_list # [t1_list, t2_list, ...], t1_list = [mu1, mu2, ...], mu1 = [x, y]
        std_list_list = [curr_std_list] + std_list_list # [t1_list, t2_list, ...], t1_list = [std1, std2, ...], std1 = [x, y]
        conf_list_list = [[1.0]*len(curr_mu_list)] + conf_list_list # [t1_list, t2_list, ...], t1_list = [conf1, conf2, ...]

        mu_list_list_msg = []
        for mu_list in mu_list_list:
            mu_list_msg = HumanTrajectoryArray()
            mu_list_msg.human_trajectories.append(HumanTrajectory(traj_points=[Point(x=float(pt[0]), y=float(pt[1])) for pt in mu_list]))
            mu_list_list_msg.append(mu_list_msg)
        self.motion_prediction_msg.mu_list_list = mu_list_list_msg

        std_list_list_msg = []
        for std_list in std_list_list:
            std_list_msg = HumanTrajectoryArray()
            std_list_msg.human_trajectories.append(HumanTrajectory(traj_points=[Point(x=float(pt[0]), y=float(pt[1])) for pt in std_list]))
            std_list_list_msg.append(std_list_msg)
        self.motion_prediction_msg.std_list_list = std_list_list_msg

        conf_list_list_msg = []
        for conf_list in conf_list_list:
            conf_list_msg = HumanTrajectoryArray()
            conf_list_msg.human_trajectories.append(HumanTrajectory(traj_points=[Point(x=float(pt), y=float(pt)) for pt in conf_list]))
            conf_list_list_msg.append(conf_list_msg)
        self.motion_prediction_msg.conf_list_list = conf_list_list_msg

        # self.get_logger().info(f"past_traj_NN: {[(round(x[0], 2), round(x[1], 2)) for x in trajs_nn[0]][-5:]}")
        # self.get_logger().info(f"mu_list: {[(round(x[0], 2), round(x[1], 2)) for x in mu_list_list[-1]]}")
        # self.get_logger().info(f"{[np.array(x, dtype=np.float64) for x in mu_list_list]}")

        self.motion_prediction_publisher.publish(self.motion_prediction_msg)
        self.motion_prediction_viz_publisher.publish(self.motion_prediction_to_vis_msg(mu_list_list, std_list_list, conf_list_list))

    def traj_callback(self, msg: HumanTrajectoryArray):
        human_trajectories: list[HumanTrajectory] = msg.human_trajectories
        self.trajs: list[list[tuple[float, float]]] = []
        for traj in human_trajectories:
            self.trajs.append([(coords.x, coords.y) for coords in traj.traj_points])
        self.traj_received = True

    def motion_prediction_to_vis_msg(
            self,
            mu_list_list: list[list[tuple[float, float]]],
            std_list_list: list[list[tuple[float, float]]],
            conf_list_list: list[list[float]],
            name_space:str="motion_prediction_ns", 
            id_start:int=0) -> MarkerArray:
        """Convert the motion prediction to a MarkerArray message"""
        marker_msg = MarkerArray()
        marker_id = id_start
        # plot the motion prediction in the shape of cylinder
        color_cylinder = (1.0, 0.0, 0.0) # red
        for _, (mu_list, std_list, conf_list) in enumerate(zip(mu_list_list, std_list_list, conf_list_list)):
            for mu, std, conf in zip(mu_list, std_list, conf_list):
                marker_id += 1
                marker_cylinder_msg = self.tuples_to_cylinder_msg(mu, std, color_cylinder, marker_id)
                marker_cylinder_msg.header.frame_id = "map"
                marker_cylinder_msg.ns = name_space
                marker_msg.markers.append(marker_cylinder_msg)
        if (marker_id - id_start) < self.last_num_markers:
            for extra_id in range(marker_id+1, id_start+self.last_num_markers+1):
                extra_marker_msg = Marker()
                extra_marker_msg.header.frame_id = "map"
                extra_marker_msg.ns = name_space
                extra_marker_msg.id = extra_id
                extra_marker_msg.action = Marker.DELETE
                marker_msg.markers.append(extra_marker_msg)
        self.last_num_markers = marker_id - id_start
        return marker_msg
    
    @staticmethod
    def tuples_to_cylinder_msg(
            mu: tuple[float, float],
            std: tuple[float, float],
            color: tuple[float, float, float],
            marker_id: int,
        ) -> Marker:
        marker_msg = Marker()
        marker_msg.id = marker_id
        marker_msg.type = Marker.CYLINDER
        marker_msg.action = Marker.ADD

        marker_msg.pose.position.x = float(mu[0])
        marker_msg.pose.position.y = float(mu[1])
        marker_msg.pose.position.z = 0.0
        marker_msg.pose.orientation.x = 0.0
        marker_msg.pose.orientation.y = 0.0
        marker_msg.pose.orientation.z = 0.0
        marker_msg.pose.orientation.w = 1.0

        marker_msg.scale.x = std[0]*3.0 + HUMAN_SIZE
        marker_msg.scale.y = std[1]*3.0 + HUMAN_SIZE
        marker_msg.scale.z = 0.1

        marker_msg.color.a = 0.2
        marker_msg.color.r = color[0]
        marker_msg.color.g = color[1]
        marker_msg.color.b = color[2]

        marker_msg.lifetime.sec = 0 # 0 is forever
        marker_msg.frame_locked = False # not locked to a frame
        return marker_msg


def main(args=None):
    rclpy.init(args=args)

    node = MotionPredictionNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
