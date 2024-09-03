import os
import math
from timeit import default_timer as timer
from typing import TypedDict, Optional, Union

import numpy as np

from configs import RPPConfiguration, CircularRobotSpecification

# This code is for the RPP trajectory algorithm

PathNode = tuple[float, float]

class DebugInfo(TypedDict):
    cost: float
    step_runtime: float

class TrajectoryTracker:
    def __init__(self, config: RPPConfiguration, robot_specification: CircularRobotSpecification, robot_id:Optional[int]=None, verbose=False):
        self.vb = verbose
        self.robot_id = robot_id
        self.config = config
        self.robot_spec = robot_specification

        # Initialization
        self._idle = True
        self.base_speed = self.robot_spec.lin_vel_max * 0.8
        self.current_velocity = 0.0
        self.current_angular_velocity =0.0

    @property
    def idle(self):
        return self._idle
    
    def load_init_states(self, current_state: np.ndarray, goal_state: np.ndarray):
        """Load the initial state and goal state.

        Args:
            current_state: Current state of the robot.
            goal_state: Final goal state of the robot (used to decelerate if close to goal).

        Raises:
            TypeError: All states should be numpy.ndarry.

        Attributes:
            state: Current state of the robot.
            final_goal: Goal state of the robot.
            past_states: List of past states of the robot.
            past_actions: List of past actions of the robot.
            solver_time_timelist: List of solver time [ms] of the robot.
            finishing: If the robot is approaching the final goal.

        Notes:
            This function sets `idle` to False.
        """
        if (not isinstance(current_state, np.ndarray)) or (not isinstance(goal_state, np.ndarray)):
            raise TypeError(f'State should be numpy.ndarry, got {type(current_state)}/{type(goal_state)}.')
        self.state = current_state
        self.final_goal = goal_state

        self.past_states: list[np.ndarray] = [current_state]
        self.past_actions: list[np.ndarray] = []
        self.solver_time_timelist: list[float] = []

        self._idle = False
        self.finishing = False # If approaching (not reaching) the last node of the reference path
    
    
    def set_current_state(self, current_state: np.ndarray):
        """To synchronize the current state of the robot with the trajectory tracker.

        Args:
            current_state: The actually current state of the robot.

        Raises:
            TypeError: The state should be numpy.ndarry.
        """
        if not isinstance(current_state, np.ndarray):
            raise TypeError(f'State should be numpy.ndarry, got {type(current_state)}.')
        self.state = current_state

    def set_ref_states(self, ref_states: np.ndarray, ref_speed:Optional[float]=None):
        """Set the local reference states for the coming time step.

        Args:
            ref_states: Local (within the horizon) reference states
            ref_speed: The reference speed. If None, use the default speed.
            
        Notes:
            This method will overwrite the base speed.
        """
        self.ref_states = ref_states
        if ref_speed is not None:
            self.base_speed = ref_speed
        else:
            self.base_speed = self.robot_spec.lin_vel_max * 0.8

    def check_termination_condition(self, external_check=True) -> bool:
        """Check if the robot finishes the trajectory tracking.

        Args:
            external_check: If this is true, the controller will check if it should terminate. Defaults to True.

        Returns:
            _idle: If the robot finishes the trajectory tracking.
        """
        if external_check:
            self.finishing = True
            if np.allclose(self.state[:2], self.final_goal[:2], atol=0.1, rtol=0) and abs(self.past_actions[-1][0]) < 0.1:
                self._idle = True
                if self.vb:
                    print(f"[{self.__class__.__name__}-{self.robot_id}] Trajectory tracking finished.")
        return self._idle
    

    def calculate_lookahead_point(self, path, position, lt, current_velocity):
        """
        Calculate the lookahead point on the path based on the current position.

        Args:
            path: Predicted Path Points
            position: Current location of the robot
            lookahead_distance: Distance according to app algorithm  Lt = vt*lt

        Returns:
            closest_point: The closest lookahead point to the robot.
        """
        lookahead_distance = current_velocity * lt

        for point in path:
            dist = np.linalg.norm(np.array(point) - np.array(position))
            if dist >= lookahead_distance:
              return point
        return path[-1] # If no point meets the lookahead distance, return the last point in the path
    
    def calculate_lookahead_point_sparse_env(self, path, position, lt, current_velocity, robot_pose):
        """Calculate the lookahead point on the path based on the current position.

        Args:
            path: Predicted Path Points
            position: Current location of the robot
            lookahead_distance: Distance according to app algorithm  Lt = vt*lt

        Returns:
            closest_point: The closest lookahead point to the robot.
        """
        lookahead_distance = current_velocity * lt
        for i in range(len(path)):
            point = path[i]
            dist = np.linalg.norm(np.array(point) - np.array(position))
            if dist >= lookahead_distance:
                if i == 0:
                    return point  # Return the first point if it meets the lookahead distance
                prev_point = path[i - 1]
                # Calculate the intersection point on the line segment if possible
                try:
                    prev_point = self.map_to_robot(prev_point, robot_pose)
                    point = self.map_to_robot(point, robot_pose)
                    intersection_point = self.circleSegmentIntersection(prev_point, point, lookahead_distance)
                    intersection_point= self.robot_to_map(intersection_point, robot_pose)
                    return intersection_point
                except ValueError:
                    # If no valid intersection is found, return the current point
                    return point
        # If no point meets the lookahead distance, return the last point in the path
        return path[-1]
    
    def calculate_curvature(self, lookahead_point, position, orientation):
        """Calculate the curvature required to reach the lookahead point.
        
        Args:
            position: Current location of the robot
            orientation: Robot orientation
        
        Returns:
            curvature: curvature required to reach the path point
        """
        dx = lookahead_point[0] - position[0]
        dy = lookahead_point[1] - position[1]
        local_x = dx * math.cos(-orientation) - dy * math.sin(-orientation)
        local_y = dx * math.sin(-orientation) + dy * math.cos(-orientation)
        
        L = math.sqrt(local_x**2 + local_y**2)
        if L < 1e-3:
           return 0.0
        curvature = 2 * local_y / (L**2)
        return curvature
        
    def calculate_velocity(self, vt, curvature, obstacle_distance, current_velocity, goal_distance):
        """Calculate the regulated velocity based on curvature and proximity to obstacles.

        Args:
            vt: constant velocity of the robot
             obstacle: Distance of the robot from the obstacle

        Returns:
            velocity: the speed required by the robot at the position
        """
        if curvature > self.config.Tk: # Adjust the speed according to the curvature
            velocity_curvature = vt
        else:
            velocity_curvature = vt /(self.config.r_min * abs(curvature))  # The curvature is supposed to be absolute. Mark my words.:)

        if obstacle_distance <= self.config.d_prox:  # Adjust speed according to proximity
            velocity_proximity = vt * (0.1* obstacle_distance / self.config.d_prox) # There's a problem with the alpha parameter setting, so you can change it right here.
        else:
            velocity_proximity = vt

        if goal_distance <= 1:
          velocity_goal = vt * (goal_distance / 1)
          velocity_goal = max(velocity_goal, 0.05)  # Ensure a minimum speed
        else:
          velocity_goal = vt

        #Limit robot speed according to robot parameters
        target_velocity = min(velocity_curvature, velocity_proximity,velocity_goal)
        max_allowed_velocity = current_velocity + self.robot_spec.lin_acc_max*self.config.ts
        min_allowed_velocity = current_velocity + self.robot_spec.lin_acc_min*self.config.ts
        target_velocity = min(target_velocity, max_allowed_velocity)
        target_velocity = max(target_velocity, min_allowed_velocity)
        
        target_velocity = max(target_velocity, 0)
        target_velocity = min(target_velocity, self.robot_spec.lin_vel_max)

        return target_velocity  # Guaranteed minimum speed


    def run_step(self, obstacles: list[PathNode], obstacle_radius: Union[float, list[float]], sparse_env:bool=False):
        p_start = self.state[:2]
        path = self.ref_states[:, :2]
        orientation = self.state[2]
        vt = self.base_speed
        min_obstacle_distance = float('inf')
        dt = self.config.ts
        lt = self.config.lt
       
        step_time_start = timer()
        if sparse_env:
            lookahead_point = self.calculate_lookahead_point_sparse_env(path, p_start, lt, self.current_velocity, self.state)
        else:
            lookahead_point = self.calculate_lookahead_point(path, p_start, lt, self.current_velocity)
        curvature = self.calculate_curvature(lookahead_point, p_start, orientation)

        for obstacle in obstacles:
            obstacle_position = np.array(obstacle[:2])
            obstacle_radius = 0.1
            obstacle_distance = np.linalg.norm(obstacle_position - p_start) - obstacle_radius-(self.robot_spec.vehicle_width)/2
            if obstacle_distance < min_obstacle_distance:
                min_obstacle_distance = obstacle_distance

        goal_distance = np.linalg.norm(np.array(self.final_goal[:2]) - np.array(self.state[:2]))
        regulated_velocity = self.calculate_velocity(vt, curvature, min_obstacle_distance,self.current_velocity,goal_distance)
        self.current_velocity = regulated_velocity
        step_runtime = timer() - step_time_start # in ms
        action_v = regulated_velocity

        #Limit robot speed according to robot parameters
        target_angular_velocity = regulated_velocity * curvature
        max_allowed_angular_velocity = self.current_angular_velocity + self.robot_spec.ang_acc_max * dt
        min_allowed_angular_velocity = self.current_angular_velocity - self.robot_spec.ang_acc_max * dt
        action_w = min(max(target_angular_velocity, min_allowed_angular_velocity), max_allowed_angular_velocity)
        action_w = max(min(action_w, self.robot_spec.ang_vel_max), -self.robot_spec.ang_vel_max)
        self.current_angular_velocity = action_w 
      
        actions = [np.array([action_v, action_w])]
        desired_states = np.copy(self.state)
        desired_states[0] += action_v * np.cos(desired_states[2]) * dt
        desired_states[1] += action_v * np.sin(desired_states[2]) * dt
        desired_states[2] += action_w * dt
        desired_states = np.array([desired_states])        
        debug_info = DebugInfo(step_runtime=step_runtime, cost=0.0)

        self.past_actions.append(actions[0])
        self.past_states.append(self.state)
        return actions, desired_states, self.ref_states, debug_info


    @staticmethod
    def circleSegmentIntersection(p1, p2, r):
        """Find the intersection point of a line segment with a circle centered at the origin.

        Args:
            p1 (Point): The starting point of the line segment.
            p2 (Point): The ending point of the line segment.
            r (float): The radius of the circle.

        Returns:
            Point: The intersection point that lies on the segment.
        """
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]

        dx = x2 - x1
        dy = y2 - y1
        dr2 = dx * dx + dy * dy
        D = x1 * y2 - x2 * y1
        # Augmentation to only return point within segment
        d1 = x1 * x1 + y1 * y1
        d2 = x2 * x2 + y2 * y2
        dd = d2 - d1

        discriminant = r * r * dr2 - D * D
        sqrt_term = math.sqrt(discriminant)
        sign_dd = math.copysign(1.0, dd)

        ix = (D * dy + sign_dd * dx * sqrt_term) / dr2
        iy = (-D * dx + sign_dd * dy * sqrt_term) / dr2
        p = [ix,iy]
        return p

    @staticmethod
    def map_to_robot(map_point, robot_pose):
        """Convert coordinates from the map frame to the robot frame.

        Args:
            map_point (tuple): The (x, y) coordinates in the map frame.
            robot_pose (tuple): The robot's pose (x, y, theta) in the map frame,
                                where theta is the orientation angle in radians.

        Returns:
            list: The (x, y) coordinates in the robot frame as a list.
        """
        map_x, map_y = map_point
        robot_x, robot_y, theta = robot_pose
        # Translate the point to the robot's position
        translated_x = map_x - robot_x
        translated_y = map_y - robot_y
        # Rotate the translated coordinates by -theta (to align with robot's heading)
        robot_x_frame = translated_x * math.cos(-theta) - translated_y * math.sin(-theta)
        robot_y_frame = translated_x * math.sin(-theta) + translated_y * math.cos(-theta)
        return [robot_x_frame, robot_y_frame]

    @staticmethod
    def robot_to_map(robot_point, robot_pose):
        """Convert coordinates from the robot frame to the map frame.

        Args:
            robot_point (tuple): The (x, y) coordinates in the robot frame.
            robot_pose (tuple): The robot's pose (x, y, theta) in the map frame,
                                where theta is the orientation angle in radians.

        Returns:
            list: The (x, y) coordinates in the map frame as a list.
        """
        robot_x_frame, robot_y_frame = robot_point
        robot_x, robot_y, theta = robot_pose
        # Rotate the point by theta to align with the map's coordinate system
        map_x_transformed = robot_x_frame * math.cos(theta) - robot_y_frame * math.sin(theta)
        map_y_transformed = robot_x_frame * math.sin(theta) + robot_y_frame * math.cos(theta)
        # Translate the point by the robot's position to map's coordinate system
        map_x = map_x_transformed + robot_x
        map_y = map_y_transformed + robot_y
        return [map_x, map_y]