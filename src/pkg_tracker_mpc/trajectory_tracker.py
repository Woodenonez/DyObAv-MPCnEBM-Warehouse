# System import
import os
import sys
import math
import warnings
import itertools
from timeit import default_timer as timer
from typing import Callable, Optional, TypedDict
# External import
import numpy as np
from scipy.spatial import ConvexHull # type: ignore
# Custom import 
from configs import MpcConfiguration, CircularRobotSpecification
from .cost_monitor import CostMonitor, MonitoredCost


PathNode = tuple[float, float]


class Solver(): # this is not found in the .so file (in ternimal: nm -D  navi_test.so)
    import opengen as og # type: ignore
    def run(self, p: list, initial_guess=None, initial_lagrange_multipliers=None, initial_penalty=None) -> og.opengen.tcp.solver_status.SolverStatus: pass


class DebugInfo(TypedDict):
    cost: float
    closest_obstacle_list: list[list[PathNode]]
    step_runtime: float
    monitored_cost: Optional[MonitoredCost]


class TrajectoryTracker:
    """Generate a smooth trajectory tracking based on the reference path and obstacle information.

    Raises:
        TypeError: The static/dynamic obstacle weights should be a list or a float/int.
        TypeError: All states should be numpy.ndarry.
        ModuleNotFoundError: If the work mode is not supported.
        RuntimeError: If the solver cannot be run.
        ModuleNotFoundError: If the solver type is not supported.

    Attributes:
        config: MPC configuration.
        robot_spec: Robot specification.

    Methods:
        load_motion_model: Load the motion model (dynamics) of the robot.
        load_init_states: Load the initial state and goal state.
        set_work_mode: Set the basic work mode (base speed and weight parameters) of the MPC solver.
        set_current_state: To synchronize the current state of the robot with the trajectory tracker.
        set_ref_states: Set the local reference states for the coming time step.
        run_step: Run one step of trajectory tracking.
        check_termination_condition: Check if the robot finishes the trajectory tracking.

    Notes:
        The solver needs to be built before running the trajectory tracking. \n
        To run the tracker: 
            1. Load motion model and init states; 
            2. Set reference path and trajectory; 
            3. Run step.
    """
    
    def __init__(self, config: MpcConfiguration, robot_specification: CircularRobotSpecification, robot_id:Optional[int]=None, use_tcp:bool=False, verbose=False):
        """Initialize the trajectory tracker.

        Args:
            config: MPC configuration.
            robot_specification: Robot specification.
            use_tcp: If the PANOC solver is called via TCP or not. Defaults to False.
            verbose: If print out debug information. Defaults to False.
        """
        print(f"[{self.__class__.__name__}-{robot_id}] Initializing robot {robot_id if robot_id is not None else 'X'}...")

        self.vb = verbose
        self.robot_id = robot_id if robot_id is not None else 'X'
        self.config = config
        self.robot_spec = robot_specification

        # Common used parameters from config
        self.ts = self.config.ts
        self.ns = self.config.ns
        self.nu = self.config.nu
        self.N_hor = self.config.N_hor
        self.solver_type = self.config.solver_type

        # Initialization
        self._idle = True
        self._mode: str = 'none'
        self._map_loaded = False
        self._obstacle_weights()
        self.set_work_mode(mode='safe', use_predefined_speed=True)

        # Monitor
        self.monitor_on = False
        self.cost_monitor = CostMonitor(self.config, self.robot_spec, verbose)
        self.cost_monitor.init_params()

        if self.config.solver_type == 'PANOC':
            self.__import_solver(use_tcp=use_tcp)

    def __import_solver(self, root_dir:str='', use_tcp:bool=False):
        """Import the PANOC solver from the build directory.

        Args:
            root_dir: The root directory of the PANOC solver. Defaults to ''.
            use_tcp: If the PANOC solver is called via TCP or not. Defaults to False.
        """

        self.use_tcp = use_tcp
        solver_path = os.path.join(root_dir, self.config.build_directory, self.config.optimizer_name)

        import opengen as og
        if not use_tcp:
            sys.path.append(solver_path)
            self.built_solver = __import__(self.config.optimizer_name) # it loads a .so (shared library object)
            self.solver:Solver = self.built_solver.solver() # Return a Solver object with run method, cannot find it though
        else: # use TCP manager to access solver
            self.mng:og.opengen.tcp.OptimizerTcpManager = og.tcp.OptimizerTcpManager(solver_path)
            self.mng.start()
            self.mng.ping() # ensure RUST solver is up and runnings

    def _obstacle_weights(self):
        """Set the weights for static and dynamic obstacles based on the configuration.

        Raises:
            TypeError: The static/dynamic obstacle weights should be a list or a float/int.

        Attributes:
            stc_weights: penalty weights for static obstacles (only useful if soft constraints activated)
            dyn_weights: penalty weights for dynamic obstacles (only useful if soft constraints activated)
        """

        if isinstance(self.config.qstcobs, list):
            self.stc_weights = self.config.qstcobs
        elif isinstance(self.config.qstcobs, (float, int)):
            self.stc_weights = [self.config.qstcobs]*self.N_hor
        else:
            raise TypeError(f'Unsupported datatype for obstacle weights, got {type(self.config.qstcobs)}.')
        if isinstance(self.config.qdynobs, list):
            self.dyn_weights = self.config.qdynobs
        elif isinstance(self.config.qdynobs, (float, int)):
            self.dyn_weights = [self.config.qdynobs]*self.N_hor
        else:
            raise TypeError(f'Unsupported datatype for obstacle weights, got {type(self.config.qdynobs)}.')

    @property
    def idle(self):
        return self._idle

    def restart_solver(self):
        if self.vb:
            print(f"[{self.__class__.__name__}-{self.robot_id}] Restarting solver...")
        
        if self.use_tcp:
            # check if self.mng exists
            if hasattr(self, 'mng'):
                self.mng.kill()
        else:
            if hasattr(self, 'solver'):
                del self.solver
                del self.built_solver
        self.__import_solver(use_tcp=self.use_tcp)


    def get_stc_constraints(self, static_obstacles: list[list[PathNode]]) -> tuple[list, list[list[PathNode]]]:
        n_stc_obs = self.config.Nstcobs * self.config.nstcobs
        stc_constraints = [0.0] * n_stc_obs
        map_obstacle_list = self.get_closest_n_stc_obstacles(static_obstacles)
        for i, map_obstacle in enumerate(map_obstacle_list):
            b, a0, a1 = self.polygon_halfspace_representation(np.array(map_obstacle))
            stc_constraints[i*self.config.nstcobs : (i+1)*self.config.nstcobs] = (b+a0+a1)
        return stc_constraints, map_obstacle_list
    
    def get_closest_n_stc_obstacles(self, static_obstacles: list[list[PathNode]]) -> list[list[PathNode]]:
        short_obs_list = []
        dists_to_obs = []
        if len(static_obstacles) <= self.config.Nstcobs:
            return static_obstacles
        for obs in static_obstacles:
            dists = self.lineseg_dists(self.state[:2], np.array(obs), np.array(obs[1:] + [obs[0]]))
            dists_to_obs.append(np.min(dists))
        selected_idc = np.argpartition(dists_to_obs, self.config.Nstcobs)[:self.config.Nstcobs]
        for i in selected_idc:
            short_obs_list.append(static_obstacles[i])
        return short_obs_list

    def get_dyn_constraints(self, full_dyn_obstacle_list=None):
        """Get the parameters for dynamic obstacle constraints from a list of dynamic obstacles.

        Args:
            full_dyn_obstacle_list: Each element contains info about one dynamic obstacle. Defaults to None.
        """
        params_per_dyn_obs = (self.config.N_hor+1) * self.config.ndynobs
        dyn_constraints = [0.0] * self.config.Ndynobs * params_per_dyn_obs
        if full_dyn_obstacle_list is not None:
            for i, dyn_obstacle in enumerate(full_dyn_obstacle_list):
                dyn_constraints[i*params_per_dyn_obs:(i+1)*params_per_dyn_obs] = list(itertools.chain(*dyn_obstacle))
        return dyn_constraints


    def load_motion_model(self, motion_model: Callable) -> None:
        """Load the motion model (dynamics) of the robot.

        Args:
            motion_model: The motion model of the robot, should be a Callable object.

        Notes:
            Form: The motion model should be `s'=f(s,a,ts)` (takes in a state and an action and returns the next state).
            Model: The motion model should be the same as the builder's motion model.
        """
        self.motion_model = motion_model
        self.cost_monitor.load_motion_model(motion_model)

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
            cost_timelist: List of cost values of the robot.
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
        self.cost_timelist: list[float] = []
        self.solver_time_timelist: list[float] = []

        self._idle = False
        self.finishing = False # If approaching (not reaching) the last node of the reference path

    def set_work_mode(self, mode:str='safe', use_predefined_speed:bool=True):
        """Set the basic work mode (base speed and weight parameters) of the MPC solver.

        Args:
            mode: Be "aligning" (start) or "safe" (20% speed) or "work" (80% speed) or "super" (full speed). Defaults to 'safe'.

        Raises:
            ModuleNotFoundError: If the mode is not supported.

        Attributes:
            base_speed: The reference speed
            tuning_params: Penalty parameters for MPC

        Notes:
            This method will overwrite the base speed.
        """
        if mode == self._mode:
            return

        if self.vb:
            print(f"[{self.__class__.__name__}-{self.robot_id}] Setting work mode to {mode}.")
        self._mode = mode
        if mode == 'aligning':
            if use_predefined_speed:
                self.base_speed = self.robot_spec.lin_vel_max*0.5
            self.tuning_params = [self.config.qpos, self.config.qvel, self.config.qtheta, self.config.lin_vel_penalty, self.config.ang_vel_penalty,
                                  self.config.qpN, self.config.qthetaN, self.config.qrpd, self.config.lin_acc_penalty, self.config.ang_acc_penalty]
            self.tuning_params = [x*0.1 for x in self.tuning_params]
            self.tuning_params[2] = max(self.tuning_params) * 100
        else:
            self.tuning_params = [self.config.qpos, self.config.qvel, self.config.qtheta, self.config.lin_vel_penalty, self.config.ang_vel_penalty,
                                  self.config.qpN, self.config.qthetaN, self.config.qrpd, self.config.lin_acc_penalty, self.config.ang_acc_penalty]
            if use_predefined_speed:
                if mode == 'safe':
                    self.base_speed = self.robot_spec.lin_vel_max*0.2
                elif mode == 'work':
                    self.base_speed = self.robot_spec.lin_vel_max*0.8
                elif mode == 'super':
                    self.base_speed = self.robot_spec.lin_vel_max*1.0
                else:
                    raise ModuleNotFoundError(f'There is no mode called {mode}.')

    def set_monitor(self, monitor_on:bool=True):
        """Set the monitor on or off.

        Args:
            monitor_on: If the monitor is on. Defaults to True.
        """
        self.monitor_on = monitor_on

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
            self.set_work_mode(mode='work', use_predefined_speed=True)

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


    def run_step(self, static_obstacles: list[list[PathNode]], full_dyn_obstacle_list:Optional[list]=None, other_robot_states:Optional[list]=None, 
                 map_updated:bool=True, report_cost:bool=False):
        """Run the trajectory planner for one step given the surrounding environment.

        Args:
            static_obstacles: A list of static obstacles, each element is a list of points (x,y).
            full_dyn_obstacle_list: A list of dynamic obstacles. Defaults to None.
            other_robot_states: A list of other robots' states. Defaults to None.
            map_updated: If the map is updated at this time step. Defaults to True.
            report_cost: If the cost should be reported. Defaults to False.

        Returns:
            actions: A list of future actions
            pred_states: A list of predicted states
            ref_states: Reference states
            debug_info: Debug information, details in Notes.

        Notes:
            cost: The cost of the predicted trajectory
            closest_obstacle_list: A list of closest static obstacles
            step_runtime: The runtime of this step
            monitored_cost: The monitored costs if the monitor is on
        """
        if self.vb and not self.monitor_on and report_cost:
            warnings.warn("The monitor is off, the cost will not be reported.", UserWarning)

        step_time_start = timer()
        if map_updated or (not self._map_loaded):
            self._stc_constraints, self._closest_obstacle_list = self.get_stc_constraints(static_obstacles)
            self._map_loaded = True
        dyn_constraints = self.get_dyn_constraints(full_dyn_obstacle_list)
        actions, pred_states, ref_states, cost, monitored_cost = self._run_step(self._stc_constraints, dyn_constraints, other_robot_states, report_cost)
        step_runtime = timer()-step_time_start
        debug_info = DebugInfo(cost=cost, 
                               closest_obstacle_list=self._closest_obstacle_list, 
                               step_runtime=step_runtime, 
                               monitored_cost=monitored_cost)
        return actions, pred_states, ref_states, debug_info

    def _run_step(self, stc_constraints: list, dyn_constraints: list, other_robot_states:Optional[list]=None, report_cost:bool=False):
        """Run the trajectory planner for one step, wrapped by `run_step`.

        Args:
            other_robot_states: A list with length "ns*N_hor*Nother" (E.x. [0,0,0] * (self.N_hor*self.config.Nother)). Defaults to None.

        Raises:
            RuntimeError: If the solver cannot be run.

        Returns:
            actions: A list of future actions
            pred_states: A list of predicted states
            ref_states: Reference states
            cost: The cost of the predicted trajectory
            monitored_costs: The monitored costs if the monitor is on
        """
        if other_robot_states is None:
            other_robot_states = [-10] * (self.ns*(self.N_hor+1)*self.config.Nother)

        ### Get reference states ###
        ref_states = self.ref_states.copy()
        finish_state = ref_states[-1,:]
        current_refs = ref_states.reshape(-1).tolist()

        ### Get reference velocities ###
        dist_to_goal = math.hypot(self.state[0]-self.final_goal[0], self.state[1]-self.final_goal[1]) # change ref speed if final goal close
        if (dist_to_goal < self.base_speed*self.N_hor*self.ts) and self.finishing:
            speed_ref = dist_to_goal / self.N_hor / self.ts * 2
            speed_ref = min(speed_ref, self.robot_spec.lin_vel_max)
            speed_ref_list = [speed_ref]*self.N_hor
        else:
            speed_ref_list = [self.base_speed]*self.N_hor

        last_u = self.past_actions[-1] if len(self.past_actions) else np.zeros(self.nu)

        ### Complementary restrictions for velocity and angular velocity ###
        # speed_decay = 0.0
        # current_ref_theta = math.degrees(ref_states[0, 2]) % 360
        # current_theta = math.degrees(self.state[2]) % 360
        # if (theta_diff := (abs(current_ref_theta - current_theta) % 180)) > 120:
        #     self.set_work_mode(mode='aligning')
        # elif theta_diff > 60:
        #     speed_decay = min(max(theta_diff/180, 0.0), 1.0)
        #     self.set_work_mode(mode='work', use_predefined_speed=False)
        # else:
        self.set_work_mode(mode='work', use_predefined_speed=False)
            
        ### Assemble parameters for solver & Run MPC###
        params = list(last_u) + list(self.state) + list(finish_state) + self.tuning_params + \
                 current_refs + speed_ref_list + other_robot_states + \
                 stc_constraints + dyn_constraints + self.stc_weights + self.dyn_weights

        try:
            # self.solver_debug(stc_constraints) # use to check (visualize) the environment
            taken_states, pred_states, actions, cost, solver_time, exit_status, u = self.run_solver(params, self.state, self.config.action_steps)
            # actions = [x*np.array([1.0-speed_decay, 1.0]) for x in actions]
            if exit_status in self.config.bad_exit_codes and self.vb:
                print(f"[{self.__class__.__name__}-{self.robot_id}] Bad converge status: {exit_status}")
        except RuntimeError:
            if self.use_tcp:
                self.mng.kill()
            raise RuntimeError(f"[{self.__class__.__name__}-{self.robot_id}] Cannot run solver.")
        
        monitored_costs = None
        if self.monitor_on:
            monitored_costs = self.cost_monitor.get_cost(self.state, params, u, report=report_cost)

        assert isinstance(cost, float)
        assert isinstance(actions, list)
        assert isinstance(pred_states, list)
        self.past_states.append(self.state)
        self.past_states += taken_states[:-1]
        self.past_actions += actions
        self.state = taken_states[-1]
        self.cost_timelist.append(cost)
        self.solver_time_timelist.append(solver_time)

        return actions, pred_states, ref_states, cost, monitored_costs

    def run_solver(self, parameters:list, state: np.ndarray, take_steps:int=1):
        """Run the solver for the pre-defined MPC problem.

        Args:
            parameters: All parameters used by MPC, defined while building.
            state: The current state.
            take_steps: The number of control step taken by the input. Defaults to 1.

        Raises:
            ModuleNotFoundError: If the solver type is not supported.

        Returns:
            taken_states: List of taken states, length equal to take_steps.
            pred_states: List of predicted states at this step, length equal to horizon N.
            actions: List of taken actions, length equal to take_steps.
            cost: The cost value of this step
            solver_time: Time cost for solving MPC of the current time step
            exit_status: The exit state of the solver.
            u: The optimal control input within the horizon.

        Notes:
            The motion model (dynamics) is defined initially.
        """
        if self.solver_type == 'PANOC':
            if self.use_tcp:
                return self.run_solver_tcp(parameters, state, take_steps)

            import opengen as og
            solution:og.opengen.tcp.solver_status.SolverStatus = self.solver.run(parameters)
            
            u:list[float]       = solution.solution
            cost:float          = solution.cost
            exit_status:str     = solution.exit_status
            solver_time:float   = solution.solve_time_ms

        elif self.solver_type == 'Casadi':
            raise NotImplementedError
            # cas_solver = CasadiSolver(self.config, self.robot_spec, parameters, self.next_initial_guess)
            # u, cost, exit_status, solver_time, next_initial_guess = cas_solver.run()
            # self.next_initial_guess = next_initial_guess

        else:
            raise ModuleNotFoundError(f'There is no solver with type {self.solver_type}.')
        
        taken_states:list[np.ndarray] = []
        for i in range(take_steps):
            state_next = self.motion_model(state, np.array(u[(i*self.nu):((i+1)*self.nu)]), self.ts)
            taken_states.append(state_next)

        pred_states:list[np.ndarray] = [taken_states[-1]]
        for i in range(len(u)//self.nu):
            pred_state_next = self.motion_model(pred_states[-1], np.array(u[(i*self.nu):(2+i*self.nu)]), self.ts)
            pred_states.append(pred_state_next)
        pred_states = pred_states[1:]

        actions = np.array(u[:self.nu*take_steps]).reshape(take_steps, self.nu).tolist()
        actions = [np.array(action) for action in actions]
        return taken_states, pred_states, actions, cost, solver_time, exit_status, u

    def run_solver_tcp(self, parameters:list, state: np.ndarray, take_steps:int=1):
        solution = self.mng.call(parameters)
        if solution.is_ok(): # Solver returned a solution
            solution = solution.get()
            u:list[float]       = solution.solution
            cost:float          = solution.cost
            exit_status:str     = solution.exit_status
            solver_time:float   = solution.solve_time_ms
        else: # Invocation failed - an error report is returned
            solver_error = solution.get()
            error_code = solver_error.code
            error_msg = solver_error.message
            self.mng.kill() # kill so rust code wont keep running if python crashes
            raise RuntimeError(f"[{self.__class__.__name__}-{self.robot_id}] MPC Solver error: [{error_code}]{error_msg}")

        taken_states:list[np.ndarray] = []
        for i in range(take_steps):
            state_next = self.motion_model( state, np.array(u[(i*self.nu):((i+1)*self.nu)]), self.ts )
            taken_states.append(state_next)

        pred_states:list[np.ndarray] = [taken_states[-1]]
        for i in range(len(u)//self.nu):
            pred_state_next = self.motion_model( pred_states[-1], np.array(u[(i*self.nu):(2+i*self.nu)]), self.ts )
            pred_states.append(pred_state_next)
        pred_states = pred_states[1:]

        actions = np.array(u[:self.nu*take_steps]).reshape(take_steps, self.nu).tolist()
        actions = [np.array(action) for action in actions]
        return taken_states, pred_states, actions, cost, solver_time, exit_status, u
    
    def report_cost(self, real_cost: float, step_runtime: float, monitored_cost: MonitoredCost, object_id:Optional[str]=None, report_steps:bool=False):
        def colored_print(r, g, b, text, end='\n'):
            print(f"\033[38;2;{r};{g};{b}m{text} \033[38;2;255;255;255m", end=end) 
        if self.monitor_on:
            self.cost_monitor.report_cost(monitored_cost, object_id=object_id, report_steps=report_steps)
        solver_time = round(self.solver_time_timelist[-1], 3)
        if solver_time > 0.8*self.ts*1000:
            colored_print(0, 255, 0, f"Mode: {self._mode}. Real cost: {round(real_cost, 4)}. Step runtime: {round(step_runtime, 3)} sec.", end=' ')
            colored_print(255, 0, 0, f"Solver time: {solver_time} ms.")
        else:
            colored_print(0, 255, 0, f"Mode: {self._mode}. Real cost: {round(real_cost, 4)}. Step runtime: {round(step_runtime, 3)} sec. Solver time: {solver_time} ms.")
        print("="*20)
        

    @staticmethod
    def lineseg_dists(points: np.ndarray, line_points_1: np.ndarray, line_points_2: np.ndarray) -> np.ndarray:
        """Cartesian distance from point to line segment.

        Args:
            p: (n_p, 2)
            a: (n_l, 2)
            b: (n_l, 2)

        Returns:
            o: (n_p, n_l)

        References:
            Link: https://stackoverflow.com/a/54442561/11208892
        """
        p, a, b = points, line_points_1, line_points_2
        if len(p.shape) < 2:
            p = p.reshape(1,2)
        n_p, n_l = p.shape[0], a.shape[0]
        # normalized tangent vectors
        d_ba = b - a
        d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))
        # signed parallel distance components, rowwise dot products of 2D vectors
        s = np.multiply(np.tile(a, (n_p,1)) - p.repeat(n_l, axis=0), np.tile(d, (n_p,1))).sum(axis=1)
        t = np.multiply(p.repeat(n_l, axis=0) - np.tile(b, (n_p,1)), np.tile(d, (n_p,1))).sum(axis=1)
        # clamped parallel distance
        h = np.amax([s, t, np.zeros(s.shape[0])], axis=0)
        # perpendicular distance component, rowwise cross products of 2D vectors  
        d_pa = p.repeat(n_l, axis=0) - np.tile(a, (n_p,1))
        c = d_pa[:, 0] * np.tile(d, (n_p,1))[:, 1] - d_pa[:, 1] * np.tile(d, (n_p,1))[:, 0]
        return np.hypot(h, c).reshape(n_p, n_l)

    @staticmethod
    def polygon_halfspace_representation(polygon_points: np.ndarray):
        """Compute the H-representation of a set of points (facet enumeration).

        Returns:
            A: (L x d). Each row in A represents hyperplane normal.
            b: (L x 1). Each element in b represents the hyperpalne constant bi
        
        References:
            Link: https://github.com/d-ming/AR-tools/blob/master/artools/artools.py
        """
        hull = ConvexHull(polygon_points)
        hull_center = np.mean(polygon_points[hull.vertices, :], axis=0)  # (1xd) vector

        K = hull.simplices
        V = polygon_points - hull_center # perform affine transformation
        A_ = np.nan * np.empty((K.shape[0], polygon_points.shape[1]))

        rc = 0
        for i in range(K.shape[0]):
            ks = K[i, :]
            F = V[ks, :]
            if np.linalg.matrix_rank(F) == F.shape[0]:
                f = np.ones(F.shape[0])
                A_[rc, :] = np.linalg.solve(F, f)
                rc += 1

        A:np.ndarray = A_[:rc, :]
        b:np.ndarray = np.dot(A, hull_center.T) + 1.0
        return b.tolist(), A[:,0].tolist(), A[:,1].tolist()