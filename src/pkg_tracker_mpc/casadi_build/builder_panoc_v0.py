from typing import Callable, TypedDict, Union, cast
from copy import deepcopy

import casadi as ca # type: ignore
from opengen import opengen as og # type: ignore # or "import opengen as og"

from . import mpc_helper as mh
from . import mpc_cost as mc

from configs import MpcConfiguration, CircularRobotSpecification


class PenaltyTerms(TypedDict):
    pos: Union[float, ca.SX]
    vel: Union[float, ca.SX]
    theta: Union[float, ca.SX]
    v: Union[float, ca.SX]
    w: Union[float, ca.SX]
    posN: Union[float, ca.SX]
    thetaN: Union[float, ca.SX]
    rpd: Union[float, ca.SX]
    acc_penalty: Union[float, ca.SX]
    w_acc_penalty: Union[float, ca.SX]


class PanocBuilder:
    """Build the MPC module via OPEN. Define states, inputs, cost, and constraints.

    Methods:
        load_motion_model: Load the motion model for the MPC problem.
        build: Build the MPC problem and solver.
    """
    _large_weight = 1000
    _small_weight = 10

    def __init__(self, mpc_config: MpcConfiguration, robot_config: CircularRobotSpecification):
        self._cfg = mpc_config
        self._spec = robot_config
        ### Frequently used
        self.ts = self._cfg.ts        # sampling time
        self.ns = self._cfg.ns        # number of states
        self.nu = self._cfg.nu        # number of inputs
        self.N_hor = self._cfg.N_hor  # control/pred horizon

        self._load_variables()

    @classmethod
    def from_yaml(cls, mpc_cfg_fpath: str, robot_cfg_fpath: str):
        mpc_config = MpcConfiguration.from_yaml(mpc_cfg_fpath)
        robot_config = CircularRobotSpecification.from_yaml(robot_cfg_fpath)
        return cls(mpc_config, robot_config)

    def _load_variables(self):
        nu = self.nu
        ns = self.ns
        N = self.N_hor

        self._u = ca.SX.sym('u', nu*N)      # 0. Inputs from 0 to N-1
        self._u_m1 = ca.SX.sym('u_m1', nu)  # 1. Input at kt=-1
        self._s_0 = ca.SX.sym('s_0', ns)    # 2. State at kt=0
        self._s_N = ca.SX.sym('s_N', ns)    # 3. State of goal at kt=N
        self._q = ca.SX.sym('q', self._cfg.nq)  # 4. Penalty for terms related to states/inputs

        self._r_s = ca.SX.sym('r_s', ns*N)  # 5. Reference states
        self._r_v = ca.SX.sym('r_v', N)     # 6. Reference speed

        self._c_0 = ca.SX.sym('c_0', ns*self._cfg.Nother)   # 7. States of other robots at kt=0
        self._c = ca.SX.sym('c', ns*N*self._cfg.Nother)     # 8. Predicted states of other robots

        self._o_s = ca.SX.sym('os', self._cfg.Nstcobs*self._cfg.nstcobs)        # 9. Static obstacles
        self._o_d = ca.SX.sym('od', self._cfg.Ndynobs*self._cfg.ndynobs*(N+1))  # 10. Dynamic obstacles
        self._q_stc = ca.SX.sym('qstc', N) # 11. Static obstacle weights
        self._q_dyn = ca.SX.sym('qdyn', N) # 12. Dynamic obstacle weights

        self._z = ca.vertcat(self._u_m1, self._s_0, self._s_N, self._q, 
                             self._r_s, self._r_v, 
                             self._c_0, self._c, 
                             self._o_s, self._o_d, self._q_stc, self._q_dyn)
        self._z = cast(ca.SX, self._z)
        self._num_params = self._z.shape[0]

        self._q_terms = PenaltyTerms(
            pos=self._q[0], 
            vel=self._q[1], 
            theta=self._q[2], 
            v=self._q[3], 
            w=self._q[4],
            posN=self._q[5], 
            thetaN=self._q[6], 
            rpd=self._q[7],
            acc_penalty=self._q[8], 
            w_acc_penalty=self._q[9]
        )

    def load_motion_model(self, motion_model: Callable[[ca.SX, ca.SX, float], ca.SX]):
        """Load the motion model for the MPC problem.

        Args:
            motion_model: Callable function `s'=f(s,u,ts)` that generates next state given the current state and action.
        """
        self._motion_model = motion_model

    def dynamic_obstacle_enlarge(self, time_step: int) -> float:
        """Enlarge the dynamic obstacle based on the time step."""
        vm = self._spec.vehicle_width/2
        sm = self._spec.social_margin
        tm = vm + sm
        em = vm*2 + sm
        enlarge_list = [0.0, 0.0, 0.0, 0.0, 0.0,
                        vm, vm, vm, vm, vm,
                        tm, tm, tm, tm, tm,
                        em, em, em, em, em]
        return enlarge_list[time_step]
                        

    def step_cost(self, step_in_horizon: int, action: ca.SX, last_state: ca.SX,
                  penalty_terms: PenaltyTerms, q_stcobs: list[ca.SX], q_dynobs: list[ca.SX],
                  ref_states: ca.SX, ref_speed: ca.SX, 
                  other_robot_positions: ca.SX, other_robot_pred_positions: ca.SX,
                  static_obstacles: ca.SX, dynamic_obstacles: ca.SX,
                  critical_step:int=100, enable_penalty=False, enable_future_enlarge=True) -> tuple[ca.SX, mc.CostTerms, list]:
        """Calculate the cost terms at the selected time step in the predictive horizon.

        Args:
            step_in_horizon: The relative time step in the predictive horizon.
            action: The action/input at the selected time step.
            last_state: The state at the previous time step.
            penalty_terms: The penalty terms for the cost function.
            q_stcobs: The weight for the static obstacle avoidance at the selected time step.
            q_dynobs: The weight for the dynamic obstacle avoidance at the selected time step.
            ref_states: The reference states at the selected and future time steps.
            ref_speed: The reference speed at the selected time step.
            other_robot_positions: The positions (2*n) of other robots at the current time step.
            other_robot_pred_positions: The predicted positions (2*n) of other robots at the selected time step.
            static_obstacles: The parameters of the static obstacles.
            dynamic_obstacles: The parameters of the dynamic obstacles.
            critical_steps: The critical time step for dynamic object avoidance. Defaults to 100.
            enable_penalty: If the penalty terms for predictive evasion are enabled. Defaults to False.
            enable_future_enlarge: If the dynamic obstacles are enlarged for future steps. Defaults to True.

        Raises:
            ValueError: Abnormal values of current_step or critical_step.

        Returns:
            state: The state at the current time step (after applying the action).
            CostTerms: The cost terms at the current time step.
            other_stuff: Other constraints or cost terms.

        Notes:
            Critical step: Used to allivate the burden of dynamic object avoidance.
            Selected time step: Relative time step in the predictive horizon, which is not the current time step.
        """
        if not 0<=step_in_horizon<self.N_hor:
            raise ValueError(f"Time step {step_in_horizon} is out of range [0, {self.N_hor-1}].")
        if not isinstance(critical_step, int) or critical_step<0:
            raise ValueError(f"Critical step {critical_step} should be at least 0.")
        kt = step_in_horizon
        state = self._motion_model(last_state, action, self.ts)
        pts = penalty_terms
        cts = mc.CostTerms()

        if enable_future_enlarge:
            enlarge_kt = kt
        else:
            enlarge_kt = 0

        ### Reference deviation costs
        cts.cost_rpd += mc.cost_refpath_deviation(state, ref_states[:2, :], weight=pts['rpd'])
        cts.cost_rvd += pts['vel'] * (action[0]-ref_speed)**2
        # cts.cost_rvd += pts['vel'] * 10 * (ca.fmax(0, action[0]-ref_speed))**2 # speeding penalty
        cts.cost_rtd += 0.0
        cts.cost_input += ca.sum1(ca.vertcat(pts['v'], pts['w']) * action**2) 

        ### Fleet collision avoidance
        safe_distance = 2*self._spec.vehicle_width+self._spec.vehicle_margin
        critical_distance = 2*self._spec.vehicle_width
        if kt < critical_step:
            cts.cost_fleet += mc.cost_fleet_collision(state[:2], other_robot_positions, 
                                                      safe_distance=critical_distance, weight=self._large_weight)
        else:
            cts.cost_fleet += 0.0
        ## Fleet collision avoidance [Predictive]
        cts.cost_fleet_pred += mc.cost_fleet_collision(state[:2], other_robot_pred_positions,
                                                       safe_distance=safe_distance, weight=self._small_weight)
        
        ### Static obstacles
        penalty_constraints_stcobs = 0.0
        for i in range(self._cfg.Nstcobs):
            eq_param = static_obstacles[i*self._cfg.nstcobs : (i+1)*self._cfg.nstcobs]
            n_edges = int(self._cfg.nstcobs / 3) # 3 means b, a0, a1
            b, a0, a1 = eq_param[:n_edges], eq_param[n_edges:2*n_edges], eq_param[2*n_edges:]
            cts.cost_stcobs += mc.cost_inside_cvx_polygon(state, b.T, a0.T, a1.T, weight=q_stcobs)
        
            inside_stc_obstacle = mh.inside_cvx_polygon(state, b.T, a0.T, a1.T)
            penalty_constraints_stcobs += ca.fmax(0, ca.vertcat(inside_stc_obstacle))

        ### Dynamic obstacles
        penalty_constraints_dynobs = 0.0
        if kt < critical_step: # (predicted) robot at the kt-th step should avoid the dynamic obstacles at their current positions
            x_dyn     = dynamic_obstacles[0::self._cfg.ndynobs*(self.N_hor+1)]
            y_dyn     = dynamic_obstacles[1::self._cfg.ndynobs*(self.N_hor+1)]
            rx_dyn    = dynamic_obstacles[2::self._cfg.ndynobs*(self.N_hor+1)] + self.dynamic_obstacle_enlarge(enlarge_kt)
            ry_dyn    = dynamic_obstacles[3::self._cfg.ndynobs*(self.N_hor+1)] + self.dynamic_obstacle_enlarge(enlarge_kt)
            As        = dynamic_obstacles[4::self._cfg.ndynobs*(self.N_hor+1)]
            alpha_dyn = dynamic_obstacles[5::self._cfg.ndynobs*(self.N_hor+1)]
            ellipse_param = [x_dyn, y_dyn, 
                             rx_dyn+self._spec.vehicle_margin+self._spec.social_margin, 
                             ry_dyn+self._spec.vehicle_margin+self._spec.social_margin, 
                             As, alpha_dyn]
            cts.cost_dynobs += mc.cost_inside_ellipses(state.T, ellipse_param, weight=self._large_weight)

            inside_dyn_obstacle = mh.inside_ellipses(state, [x_dyn, y_dyn, rx_dyn, ry_dyn, As])
            penalty_constraints_dynobs += ca.fmax(0, inside_dyn_obstacle)
        else:
            cts.cost_dynobs += 0.0
        ### Dynamic obstacles [Predictive]
        ### (x, y, rx, ry, angle, alpha) for obstacle 0 for N steps, then similar for obstalce 1 for N steps...
        x_dyn     = dynamic_obstacles[(kt+1)*self._cfg.ndynobs  ::self._cfg.ndynobs*(self.N_hor+1)]
        y_dyn     = dynamic_obstacles[(kt+1)*self._cfg.ndynobs+1::self._cfg.ndynobs*(self.N_hor+1)]
        rx_dyn    = dynamic_obstacles[(kt+1)*self._cfg.ndynobs+2::self._cfg.ndynobs*(self.N_hor+1)] + self.dynamic_obstacle_enlarge(enlarge_kt)
        ry_dyn    = dynamic_obstacles[(kt+1)*self._cfg.ndynobs+3::self._cfg.ndynobs*(self.N_hor+1)] + self.dynamic_obstacle_enlarge(enlarge_kt)
        As        = dynamic_obstacles[(kt+1)*self._cfg.ndynobs+4::self._cfg.ndynobs*(self.N_hor+1)]
        alpha_dyn = dynamic_obstacles[(kt+1)*self._cfg.ndynobs+5::self._cfg.ndynobs*(self.N_hor+1)]
        ellipse_param = [x_dyn, y_dyn, 
                         rx_dyn+self._spec.vehicle_margin, 
                         ry_dyn+self._spec.vehicle_margin, 
                         As, alpha_dyn]
        cts.cost_dynobs_pred += mc.cost_inside_ellipses(state.T, ellipse_param, weight=q_dynobs)

        inside_dyn_obstacle = mh.inside_ellipses(state.T, [x_dyn, y_dyn, rx_dyn, ry_dyn, As])
        penalty_constraints_dynobs_pred = ca.fmax(0, inside_dyn_obstacle)

        if enable_penalty:
            other_stuff = [penalty_constraints_stcobs, penalty_constraints_dynobs, penalty_constraints_dynobs_pred]
        else:
            other_stuff = [penalty_constraints_stcobs, penalty_constraints_dynobs, 0.0]
        return state, cts, other_stuff

    def build(self, use_tcp:bool=False, test:bool=False):
        """Build the MPC problem and solver, including states, inputs, cost, and constraints.

        Args:
            use_tcp : If the solver will be called directly or via TCP.
            test : If the function is called for testing purposes, i.e. without building the solver.

        Notes:
            Inputs (u): speed, angular speed
            states (s): x, y, theta
            Shooting method: Single shooting
            
        References:
            Ellipse definition: [https://math.stackexchange.com/questions/426150/what-is-the-general-equation-of-the-ellipse-that-is-not-in-the-origin-and-rotate]
        """
        print(f'[{self.__class__.__name__}] Building MPC module...')
        
        ref_states = ca.reshape(self._r_s, (self.ns, self.N_hor)) # each column is a state
        ref_states = ca.horzcat(ref_states, ref_states[:,[-1]])

        other_x_0 = self._c_0[ ::self.ns] # first  state
        other_y_0 = self._c_0[1::self.ns] # second state
        other_robots_0 = ca.hcat([other_x_0, other_y_0]).T # every column is a state of a robot

        cost = 0.0
        penalty_constraints = 0.0
        state = deepcopy(self._s_0)
        for kt in range(0, self.N_hor): # LOOP OVER PREDICTIVE HORIZON
            action = self._u[kt*self.nu:(kt+1)*self.nu]
            other_robots_x = self._c[kt*self.ns  ::self.ns*self.N_hor] # first  state
            other_robots_y = self._c[kt*self.ns+1::self.ns*self.N_hor] # second state
            other_robots = ca.hcat([other_robots_x, other_robots_y]).T # every column is a state of a robot
            state, step_cost_terms, penalty_const_list = self.step_cost(kt, action=action, last_state=state,
                                                                        penalty_terms=self._q_terms, q_stcobs=self._q_stc[kt], q_dynobs=self._q_dyn[kt],
                                                                        ref_states=ref_states[:, kt:], ref_speed=self._r_v[kt], 
                                                                        other_robot_positions=other_robots_0, other_robot_pred_positions=other_robots,
                                                                        static_obstacles=self._o_s, dynamic_obstacles=self._o_d, critical_step=10,
                                                                        enable_penalty=True, enable_future_enlarge=False)
            cost += step_cost_terms.sum()
            penalty_constraints += sum(penalty_const_list)

        ### Terminal cost
        cost += self._q_terms['posN']*((state[0]-self._s_N[0])**2 + (state[1]-self._s_N[1])**2) + self._q_terms['thetaN']*(state[2]-self._s_N[2])**2 # terminated cost

        ### Max speed bound
        umin = [self._spec.lin_vel_min, -self._spec.ang_vel_max] * self.N_hor
        umax = [self._spec.lin_vel_max,  self._spec.ang_vel_max] * self.N_hor
        bounds = og.constraints.Rectangle(umin, umax)

        ### Acceleration bounds and cost
        v = self._u[0::2] # velocity
        w = self._u[1::2] # angular velocity
        acc   = (v-ca.vertcat(self._u_m1[0], v[0:-1]))/self.ts
        w_acc = (w-ca.vertcat(self._u_m1[1], w[0:-1]))/self.ts
        acc_constraints = ca.vertcat(acc, w_acc)
        # Acceleration bounds
        acc_min   = [ self._spec.lin_acc_min] * self.N_hor 
        w_acc_min = [-self._spec.ang_acc_max] * self.N_hor
        acc_max   = [ self._spec.lin_acc_max] * self.N_hor
        w_acc_max = [ self._spec.ang_acc_max] * self.N_hor
        acc_bounds = og.constraints.Rectangle(acc_min + w_acc_min, acc_max + w_acc_max)
        # Accelerations cost
        cost += ca.mtimes(acc.T, acc)*self._q_terms['acc_penalty']
        cost += ca.mtimes(w_acc.T, w_acc)*self._q_terms['w_acc_penalty']

        problem = og.builder.Problem(self._u, self._z, cost) \
            .with_constraints(bounds) \
            .with_aug_lagrangian_constraints(acc_constraints, acc_bounds)
        if not isinstance(penalty_constraints, float):
            problem.with_penalty_constraints(penalty_constraints)

        build_config = og.config.BuildConfiguration() \
            .with_build_directory(self._cfg.build_directory) \
            .with_build_mode(self._cfg.build_type)
        if not use_tcp:
            build_config.with_build_python_bindings()
        else:
            build_config.with_tcp_interface_config()

        meta = og.config.OptimizerMeta() \
            .with_optimizer_name(self._cfg.optimizer_name)

        solver_config = og.config.SolverConfiguration() \
            .with_initial_penalty(10) \
            .with_max_duration_micros(self._cfg.max_solver_time)
            # initial penalty = 1
            # tolerance = 1e-4
            # max_inner_iterations = 500 (given a penalty factor)
            # max_outer_iterations = 10  (increase the penalty factor)
            # penalty_weight_update_factor = 5.0
            # max_duration_micros = 5_000_000 (5 sec)

        builder = og.builder.OpEnOptimizerBuilder(problem, meta, build_config, solver_config) \
            .with_verbosity_level(1)
        if test:
            print(f"[{self.__class__.__name__}] MPC builder is tested without building.")
            return 1
        else:
            builder.build()

        print(f'[{self.__class__.__name__}] MPC module built with {self._num_params} parameters.')




