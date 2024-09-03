from typing import Callable, Optional, TypedDict
from timeit import default_timer as timer

import casadi as ca # type: ignore

from .casadi_build import mpc_cost as mc
from .casadi_build.builder_panoc import PanocBuilder, PenaltyTerms
from .casadi_build.mpc_cost import CostTerms

from configs import MpcConfiguration, CircularRobotSpecification


class MonitoredCost(TypedDict):
    total_cost: CostTerms
    step_cost_list: list[CostTerms]
    terminal_cost: float
    cost_acc: float
    cost_w_acc: float


class CostMonitor:
    """Cost monitor for MPC.

    Notes:
        Set motion model first. Every step set parameters first.
    """
    def __init__(self, mpc_config: MpcConfiguration, robot_config: CircularRobotSpecification, verbose=False) -> None:
        self._cfg = mpc_config
        self._spec = robot_config
        self.vb = verbose

        self._builder = PanocBuilder(self._cfg, self._spec)
        self.init_params()

    def init_params(self):
        if self.vb:
            print(f"[{self.__class__.__name__}] Initializing parameters...")

        def cszv(n: int) -> ca.SX: return ca.SX.zeros(n, 1)

        self._u_m1 = cszv(self._cfg.nu) # 1. Input at kt=-1
        self._s_0 = cszv(self._cfg.ns)  # 2. State at kt=0
        self._s_N = cszv(self._cfg.ns)  # 3. State at kt=N
        self._q = cszv(self._cfg.nq)    # 4. State weights

        self._r_s = cszv(self._cfg.ns*self._cfg.N_hor)  # 5. Reference states
        self._r_v = cszv(self._cfg.N_hor)               # 6. Reference inputs

        self._c_0 = cszv(self._cfg.ns*self._cfg.Nother)                 # 7. States of other robots at kt=0
        self._c = cszv(self._cfg.ns*self._cfg.N_hor*self._cfg.Nother)   # 8. Predicted states of other robots

        self._o_s = cszv(self._cfg.Nstcobs*self._cfg.nstcobs)                      # 9. Static obstacles
        self._o_d = cszv(self._cfg.Ndynobs*self._cfg.ndynobs*(self._cfg.N_hor+1))  # 10. Dynamic obstacles
        self._q_stc = cszv(self._cfg.N_hor)                                        # 11. Static obstacle weights
        self._q_dyn = cszv(self._cfg.N_hor)                                        # 12. Dynamic obstacle weights

        self._z = [
            self._u_m1, self._s_0, self._s_N, self._q, 
            self._r_s, self._r_v, self._c_0, self._c, 
            self._o_s, self._o_d, self._q_stc, self._q_dyn
        ]

        self._num_params = self._builder._num_params

    @classmethod
    def from_yaml(cls, cfg_mpc_fpath: str, cfg_robot_fpath) -> "CostMonitor":
        return cls(MpcConfiguration.from_yaml(cfg_mpc_fpath), CircularRobotSpecification.from_yaml(cfg_robot_fpath))
    
    def load_motion_model(self, motion_model: Callable):
        self._builder.load_motion_model(motion_model)

    def _set_params(self, new_params: list, new_actions: list) -> None:
        if len(new_params) != self._num_params:
            raise ValueError(f"[{self.__class__.__name__}] Number of parameters should be {self._num_params}, but got {len(new_params)}")

        accumulative_len = 0
        for params in self._z:
            param_len = params.shape[0]
            params[:] = new_params[accumulative_len:accumulative_len+param_len]
            accumulative_len += param_len
        self._u = ca.SX(new_actions)

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

        self.ref_states = ca.reshape(self._r_s, (self._cfg.ns, self._cfg.N_hor))
        self.ref_states = ca.horzcat(self.ref_states, self.ref_states[:,[-1]])[:2, :]
        other_x_0 = self._c_0[ ::self._cfg.ns] # first  state
        other_y_0 = self._c_0[1::self._cfg.ns] # second state
        self.other_robots_0 = ca.hcat([other_x_0, other_y_0]).T

    def _get_step_cost(self, kt: int, last_state: ca.SX):
        u_t = self._u[kt*self._cfg.nu:(kt+1)*self._cfg.nu]
        other_robots_x = self._c[kt*self._cfg.ns  ::self._cfg.ns*self._cfg.N_hor] # first  state
        other_robots_y = self._c[kt*self._cfg.ns+1::self._cfg.ns*self._cfg.N_hor] # second state
        other_robots = ca.hcat([other_robots_x, other_robots_y]).T # every column is a state of a robot
        state, step_cost, *_ = self._builder.step_cost(kt, action=u_t, last_state=last_state, 
                                                       penalty_terms=self._q_terms, q_stcobs=self._q_stc[kt], q_dynobs=self._q_dyn[kt],
                                                       ref_states=self.ref_states[:, kt:], ref_speed=self._r_v[kt],
                                                       other_robot_positions=self.other_robots_0, other_robot_pred_positions=other_robots,
                                                       static_obstacles=self._o_s, dynamic_obstacles=self._o_d)
        return state, step_cost
    
    def get_cost(self, last_state: ca.SX, new_params: list, new_actions: list, report=True):
        """Get the cost summary.

        Args:
            last_state: The state at kt=-1.
            new_params: All parameters at this step.
            new_actions: The actions within the horizon.

        Returns:
            MonitoredCost: The cost summary.

        Notes:
            total_cost: The total cost of this step.
            step_cost_list: A list of step costs.
            terminal_cost: The terminal cost.
            cost_acc: The acceleration cost.
            cost_w_acc: The angular acceleration cost.
        """
        start_time = timer()
        self._set_params(new_params, new_actions)
        step_cost_list:list[CostTerms] = []
        total_cost = CostTerms()
        state = last_state
        for kt in range(self._cfg.N_hor):
            state, step_cost = self._get_step_cost(kt, ca.SX(state))
            step_cost_list.append(step_cost)
            total_cost += step_cost
        terminal_cost = self._q_terms['posN']*((state[0]-self._s_N[0])**2 + (state[1]-self._s_N[1])**2) + self._q_terms['thetaN']*(state[2]-self._s_N[2])**2 # terminated cost
        terminal_cost = float(terminal_cost)

        v = self._u[0::2] # velocity
        w = self._u[1::2] # angular velocity
        acc   = (v-ca.vertcat(self._u_m1[0], v[0:-1]))/self._cfg.ts
        w_acc = (w-ca.vertcat(self._u_m1[1], w[0:-1]))/self._cfg.ts
        cost_acc = ca.mtimes(acc.T, acc)*self._q_terms['acc_penalty']
        cost_w_acc = ca.mtimes(w_acc.T, w_acc)*self._q_terms['w_acc_penalty']
        self.runtime = timer()-start_time
        monitored_cost = MonitoredCost(total_cost=total_cost, 
                                       step_cost_list=step_cost_list, 
                                       terminal_cost=terminal_cost, 
                                       cost_acc=cost_acc, 
                                       cost_w_acc=cost_w_acc)
        if report:
            self.report_cost(monitored_cost)
        return monitored_cost
    
    def report_cost(self, monitored_cost: MonitoredCost, object_id:Optional[str]=None, report_steps=False):
        total_cost = monitored_cost['total_cost']
        step_cost_list = monitored_cost['step_cost_list']
        terminal_cost = monitored_cost['terminal_cost']
        cost_acc = monitored_cost['cost_acc']
        cost_w_acc = monitored_cost['cost_w_acc']
        if object_id is not None:
            prt_obj_info = f" for {object_id}"
        else:
            prt_obj_info = ""

        final_cost = total_cost.sum_values()+terminal_cost+cost_acc+cost_w_acc
        print("-"*20)
        print(f"Cost report{prt_obj_info} - Runtime {round(self.runtime, 3)} sec - Total cost {round(float(final_cost), 4)}:")
        print(f"  - Ref path deviation: {total_cost.cost_rpd}")
        print(f"  - Ref velocity deviation: {total_cost.cost_rvd}")
        print(f"  - Input cost: {total_cost.cost_input}")
        print(f"  - Fleet collision: {total_cost.cost_fleet}")
        print(f"  - Fleet collision [Predictive]: {total_cost.cost_fleet_pred}")
        print(f"  - Static obstacle: {total_cost.cost_stcobs}")
        print(f"  - Dynamic obstacle: {total_cost.cost_dynobs}")
        print(f"  - Dynamic obstacle [Predictive]: {total_cost.cost_dynobs_pred}")
        print(f"  - Terminal cost: {terminal_cost}")
        print(f"  - Acceleration cost: {cost_acc}")
        print(f"  - Angular acceleration cost: {cost_w_acc}")

        if report_steps:
            print(f"Step cost report:")
            for i, step_cost in enumerate(step_cost_list):
                print(f"  Step {i}:")
                print(f"    - Ref path deviation: {step_cost.cost_rpd}")
                print(f"    - Ref velocity deviation: {step_cost.cost_rvd}")
                print(f"    - Input cost: {step_cost.cost_input}")
                print(f"    - Fleet collision: {step_cost.cost_fleet}")
                print(f"    - Fleet collision [Predictive]: {step_cost.cost_fleet_pred}")
                print(f"    - Static obstacle: {step_cost.cost_stcobs}")
                print(f"    - Dynamic obstacle: {step_cost.cost_dynobs}")
                print(f"    - Dynamic obstacle [Predictive]: {step_cost.cost_dynobs_pred}")


if __name__ == "__main__":
    pass
