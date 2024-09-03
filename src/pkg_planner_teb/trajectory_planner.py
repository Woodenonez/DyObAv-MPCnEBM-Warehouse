import os
from timeit import default_timer as timer
from typing import TypedDict, Optional, Union

import numpy as np
import casadi as ca # type: ignore

from configs import TebConfiguration, CircularRobotSpecification


PathNode = tuple[float, float]


class DebugInfo(TypedDict):
    cost: float
    step_runtime: float


class TrajectoryPlanner:
    def __init__(self, config: TebConfiguration, robot_specification: CircularRobotSpecification, safe_factor:float=1.0, safe_margin:float=0.0, verbose:bool=False):
        """
        Notes:
            The `safe_factor` and `safe_margin` are used to adjust the safety margin of the obstacles.
            (r_obs = safe_factor * r_obs + safe_margin)
        """
        self.vb = verbose
        self.config = config
        self.robot_spec = robot_specification

        # Initialization
        self.base_speed = self.robot_spec.lin_vel_max * 0.8

        # Build problem
        self._build_problem(safe_factor, safe_margin)
    
    def _build_problem(self, safe_factor=1.0, safe_margin=0.0):
        """
        Attributes:
            nlp: CasADi NLP dictionary
            solver: CasADi solver
            lbg: Lower bound of constraints
            ubg: Upper bound of constraints
        """
        num_steps = self.config.N_hor
        num_obstacles = self.config.N_obs
        v_max = self.robot_spec.lin_vel_max

        # Variables
        x = ca.MX.sym('x', num_steps)
        y = ca.MX.sym('y', num_steps)
        opt_vars = ca.vertcat(x, y)

        # Parameters
        p_s = ca.MX.sym('p_s', 2) # start point
        p_g = ca.MX.sym('p_g', 2) # goal point
        i_x = ca.MX.sym('i_x', num_steps) # initial x
        i_y = ca.MX.sym('i_y', num_steps) # initial y
        i_dt = ca.MX.sym('i_dt', num_steps-1) # initial dt
        o_x = ca.MX.sym('o_x', num_obstacles) # obstacle x
        o_y = ca.MX.sym('o_y', num_obstacles) # obstacle y
        o_r = ca.MX.sym('o_r', num_obstacles) # obstacle radius
        v_ref = ca.MX.sym('v_ref', 1) # reference speed
        P = ca.vertcat(p_s, p_g, i_x, i_y, i_dt, o_x, o_y, o_r, v_ref)

        # Cost
        cost = 0
        # Obstacle cost
        # for i in range(num_steps):
        #     for j in range(num_obstacles):
        #         cost += ca.fmax(0, ((2*o_r[j])**2 - ((x[i] - o_x[j])**2 + (y[i] - o_y[j])**2)))
        # Time cost
        # cost += ca.sum1( (ca.sqrt(((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)) - v_ref*i_dt)**2 )
        cost += ca.sum1( (x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2 - (v_ref*i_dt)**2 )
        # Deviation cost
        cost += 0.1 * ca.sum1((x - i_x)**2 + (y - i_y)**2)

        # Constraints
        g_eq = []
        g_eq.append(x[0] - p_s[0])
        g_eq.append(y[0] - p_s[1])
        g_eq.append(x[-1] - p_g[0])
        g_eq.append(y[-1] - p_g[1])

        g_ineq = [] # <= 0
        for i in range(num_steps-1):
            g_ineq.append((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2 - (v_max*i_dt[i])**2)
        for i in range(num_steps):
            for j in range(num_obstacles):
                g_ineq.append((safe_factor*o_r[j]+safe_margin)**2 - ((x[i] - o_x[j])**2 + (y[i] - o_y[j])**2))

        g = ca.vertcat(*g_eq, *g_ineq)

        lbg_eq = np.zeros(len(g_eq))
        lbg_ineq = -np.inf * np.ones(len(g_ineq))
        self._lbg = np.concatenate([lbg_eq, lbg_ineq])
        self._ubg = np.zeros(g.shape[0])

        self._nlp = {'x': opt_vars, 'f': cost, 'g': g, 'p': P}
        opts = {
            'ipopt.print_level': 0,        # Disable IPOPT solver verbosity
            'print_time': False,           # Disable Casadi's internal print times
            'verbose': False               # Disable general verbose output
        }
        self.solver = ca.nlpsol('solver', 'ipopt', self._nlp, opts)

        if self.vb:
            print(f"[{self.__class__.__name__}] #variables: {opt_vars.shape[0]}, #parameters: {P.shape[0]}")
            
    def _get_obs_constraints(self, obstacles: list[PathNode], obstacle_radius: Union[float, list[float]]):
        """Given positions and radii of obstacles, return the constraints for the optimization problem.
        
        Returns:
            obstacle_xs: x-coordinates of obstacles
            obstacle_ys: y-coordinates of obstacles
            obstacle_radius: radii of obstacles
        """
        if obstacles is None:
            obstacle_xs = [0] * self.config.N_obs
            obstacle_ys = [0] * self.config.N_obs
            obstacle_radius = [0] * self.config.N_obs
            return obstacle_xs, obstacle_ys, obstacle_radius
        
        if isinstance(obstacle_radius, float):
            obstacle_radius = [obstacle_radius] * len(obstacles)
        assert isinstance(obstacle_radius, list)
        obstacle_xs = [obs[0] for obs in obstacles]
        obstacle_ys = [obs[1] for obs in obstacles]
        if len(obstacles) < self.config.N_obs:
            obstacle_xs += [0] * (self.config.N_obs - len(obstacles))
            obstacle_ys += [0] * (self.config.N_obs - len(obstacles))
            obstacle_radius += [0] * (self.config.N_obs - len(obstacles))
        elif len(obstacles) > self.config.N_obs:
            obstacle_xs = obstacle_xs[:self.config.N_obs]
            obstacle_ys = obstacle_ys[:self.config.N_obs]
            obstacle_radius = obstacle_radius[:self.config.N_obs]
        return obstacle_xs, obstacle_ys, obstacle_radius
    

    def set_ref_states(self, current_state: np.ndarray, ref_states: np.ndarray, ref_speed:Optional[float]=None):
        """Set the local reference states for the coming time step.

        Args:
            ref_states: Local (within the horizon) reference states
            ref_speed: The reference speed. If None, use the default speed.
            
        Notes:
            This method will overwrite the base speed.
        """
        if not isinstance(current_state, np.ndarray):
            raise TypeError(f'State should be numpy.ndarry, got {type(current_state)}.')
        self.state = current_state

        self.ref_states = ref_states
        if ref_speed is not None:
            self.base_speed = ref_speed
        else:
            self.base_speed = self.robot_spec.lin_vel_max * 0.8

    def run_step(self, obstacles: list[PathNode], obstacle_radius: Union[float, list[float]]):
        obs_x, obs_y, obs_r = self._get_obs_constraints(obstacles, obstacle_radius)
        p_start = self.state[:2]
        p_goal = self.ref_states[-1, :2]
        x_init = self.ref_states[:, 0]
        y_init = self.ref_states[:, 1]
        dt_init = np.ones(self.config.N_hor-1) * (np.linalg.norm(p_goal - p_start) / (self.config.N_hor-1) / self.base_speed)

        x0 = np.concatenate([x_init, y_init])
        params = np.concatenate(
            [p_start[:2], p_goal[:2], x_init, y_init, dt_init, 
             obs_x, obs_y, obs_r, np.array([self.base_speed])]
        )

        step_time_start = timer()
        sol = self.solver(x0=x0, lbg=self._lbg, ubg=self._ubg, p=params)
        step_runtime = timer() - step_time_start # in ms

        cost = float(sol['f'])
        x_sol = sol['x'].full().flatten()
        x_opt = x_sol[:self.config.N_hor]
        y_opt = x_sol[self.config.N_hor:2*self.config.N_hor]
        dist_to_goal = np.linalg.norm(np.vstack([x_opt[:-1], y_opt[:-1]]).T - np.array([x_opt[-1], y_opt[-1]]), axis=1)
        theta_opt = np.arctan2(np.array(y_opt[1:])-np.array(y_opt[:-1]) + 1e-6, 
                               np.array(x_opt[1:])-np.array(x_opt[:-1]) + 1e-6)
        if any(dist_to_goal < 0.1):
            last_valid_theta = theta_opt[dist_to_goal >= 0.1][-1]
            theta_opt[dist_to_goal < 0.1] = last_valid_theta
        desired_states = np.vstack([x_opt[1:], y_opt[1:], theta_opt]).T # (N-1)*3
        current_and_desired = np.vstack([self.state, desired_states])
        debug_info = DebugInfo(cost=cost, step_runtime=step_runtime)

        return current_and_desired, debug_info

