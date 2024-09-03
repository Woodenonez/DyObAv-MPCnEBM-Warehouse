
from typing import Callable

import numpy as np
import casadi as cs # type: ignore

from . import mpc_helper
from . import mpc_cost

from basic_casadi.direct_multiple_shooting import MultipleShootingSolver
from configs import MpcConfiguration, CircularRobotSpecification


class CasadiBuilder:
    def __init__(self, config: MpcConfiguration, robot_specification: CircularRobotSpecification):
        # Define control parameters 
        self.config = config
        self.robot_spec = robot_specification
        ### Frequently used
        self.ts = self.config.ts        # sampling time
        self.ns = self.config.ns        # number of states
        self.nu = self.config.nu        # number of inputs
        self.N_hor = self.config.N_hor  # control/pred horizon

    def build(self, motion_model: Callable[[cs.SX, cs.SX, float], cs.SX]):

        print(f'[{self.__class__.__name__}] Building MPC module...')

        ### Define all parameters used by Casadi solver
        u = cs.SX.sym('u', self.nu*(self.N_hor+1))  # 0. Inputs from -1 to N_hor-1
        u_m1 = u[:self.nu]                          # 0. Previous input (at kt=-1)
        s = cs.SX.sym('s', self.ns*(self.N_hor+1))  # 1. States from 0 to N_hor
        s_0 = s[:self.ns]                           # 1. Initial state
        s_N = s[self.ns*self.N_hor:]                # 1. Final state
        r_s = cs.SX.sym('r_s', self.ns*self.N_hor)  # 2. Reference states
        r_v = cs.SX.sym('r_v', self.N_hor)          # 3. Reference speed
        q_bas = cs.SX.sym('q', self.config.nq)      # 4. Basic penalty parameters

        c = cs.SX.sym('c',                      # 5. States of other robots at kt=0 to N_hor
                      self.ns*self.config.Nother*(self.N_hor+1))
        c_0 = c[:self.ns*self.config.Nother]    # 5. Current states of other robots     
        c_p = c[self.ns*self.config.Nother:]    # 5. Predicted states of other robots
        q_rob = cs.SX.sym('q_rob', self.N_hor)  # 6. Penalty parameters for other robots

        o_s = cs.SX.sym('o_s',  # 7. Static obstacles
                        self.config.nstcobs*self.config.Nstcobs)
        o_d = cs.SX.sym('o_d',  # 8. Dynamic obstacles (current + predicted)
                        self.config.ndynobs*self.config.Ndynobs*(self.N_hor+1))
        q_stc = cs.SX.sym('q_stc',  self.N_hor) # 9. Penalty parameters for static obstacles
        q_dyn = cs.SX.sym('q_dyn',  self.N_hor) # 10. Penalty parameters for dynamic obstacles

        ### NOTE: The first ns parameters must be the initial state
        z:cs.SX = cs.vertcat(s, r_s, u, r_v, q_bas, c, q_rob, o_s, o_d, q_stc, q_dyn) # all parameters

        ### Distribute parameters
        (x0, y0, theta0) = (s_0[0], s_0[1], s_0[2])
        (xN, yN, thetaN) = (s_N[0], s_N[1], s_N[2])
        (v_init, w_init) = (u_m1[0], u_m1[1])

        q_X1, q_X2 = q[0], q[1] # XXX Find out what these are

        ref_states = cs.reshape(r_s, (self.ns, self.N_hor))      # col vector
        ref_states = cs.horzcat(ref_states, ref_states[:, [-1]]) # repeat the last ref_state

        cost = 0
        penalty_constraints = 0
        state = cs.vertcat(x0, y0, theta0)



        ms_solver = MultipleShootingSolver(self.ns, self.nu, self.ts, self.N_hor)
        ms_solver.set_motion_model(motion_model)
        ms_solver.set_parameters(z, with_initial_state=True)


    
    def return_step_cost(self, 
                         state: cs.SX, ref_path: cs.SX, q_rpd: cs.SX,
                         input: cs.SX, ref_input: cs.SX, q_vel: cs.SX, r_v: cs.SX, r_w: cs.SX,
                         static_obstacle_parameters: cs.SX, q_static_obstacle: cs.SX,
                         dynamic_obstacle_parameters: cs.SX, q_dynamic_obstacle: cs.SX,
                         other_robot_current_states: cs.SX, other_robot_future_states: cs.SX):
        """Return the cost of a single step in the horizon

        Notes:
            All arguments are based on column vectors.
        """
        step_cost = 0
        ### Reference deviation costs
        step_cost += mpc_cost.cost_refstate_deviation(state, ref_path, q_rpd)
        step_cost += mpc_cost.cost_refvalue_deviation(input[0], ref_input[0], q_vel)
        step_cost += mpc_cost.cost_control_actions(input.T, cs.horzcat(r_v, r_w))
        ### Fleet collision avoidance
        other_x_0 = other_robot_current_states[ ::self.ns] # first  state
        other_y_0 = other_robot_current_states[1::self.ns] # second state
        other_robots_0 = cs.hcat([other_x_0, other_y_0]) # states of other robots at time 0
        other_robots_0 = cs.transpose(other_robots_0) # every column is a state of a robot
        step_cost += mpc_cost.cost_fleet_collision(state[:2].T, other_robots_0.T, 
                                                   safe_distance=2*(self.robot_spec.vehicle_width+self.robot_spec.vehicle_margin), weight=1000)
        ### Fleet collision avoidance [Predictive]
        other_robots_x = other_robot_future_states[kt*self.ns  ::self.ns*self.N_hor] # first  state
        other_robots_y = other_robot_future_states[kt*self.ns+1::self.ns*self.N_hor] # second state
        other_robots = cs.hcat([other_robots_x, other_robots_y]) # states of other robots at time kt (Nother*ns)
        other_robots = cs.transpose(other_robots) # every column is a state of a robot
        step_cost += mpc_cost.cost_fleet_collision(state[:2].T, other_robots.T, 
                                                   safe_distance=2*(self.robot_spec.vehicle_width+self.robot_spec.vehicle_margin), weight=10)


    
    def individual_cost(self, Cost_dict: dict, varialbes: cs.SX, state: list, input: list):
        #Get a list of 3 states and 2 inputs for N
        merge_list = state[:3]
        for i in range(self.N_hor):
            merge_list += input[2*(i):2*(i)+2]
            merge_list += state[3*(i+1):3*(i+1)+3]
        #Calc individual costs
        for key, value in Cost_dict.items():
            cost_ind = cs.Function(key,[varialbes],[value])
            print(key,' : ', cost_ind(merge_list))
        #ref_path_cost = cs.Function('test',[varialbes],[Cost_dict['ref_path']])
        #print('ref_path',ref_path_cost(merge_list))
    
    def run(self):
        """Run the Casadi solver for the entire horizon

        Returns:
            Any: A list of optimal control inputs, calulated total cost, 
            the exit status of the solver and the time it took to run the solver
        """
        # Define a symbolic continious function
        fk = self.return_continuous_function()

        # Discretize the continious function 
        ms_solver = MultipleShootingSolver(self.ns, self.nu, self.ts, self.N_hor, self.config,self.robot_spec,self.initial_guess)
        ms_solver.set_initial_state(self.x0)
        ms_solver.set_motion_model(self.unicycle_model,c2d=False)
        ms_solver.set_parameters(self.params)
        
        map = self.get_map_data()
        bounds = map['boundary_coords']
        # Define state and output bounds
        ms_solver.set_control_bound([self.lin_vel_min, self.ang_vel_min], [self.lin_vel_max, self.ang_vel_max])
        ms_solver.set_state_bound([[bounds[0][0],bounds[0][1],-2*cs.pi]]*(self.N_hor+1), 
                                [[bounds[2][0],bounds[2][1],2*cs.pi]]*(self.N_hor+1))
        ms_solver.set_stcobs_constraints(map['obstacle_list'])
    
        
        problem, Cost_dict =  ms_solver.build_problem()

        sol, solver_time, exit_status, solver_cost = ms_solver.solve()
        print('Total Cost: ', solver_cost)
     
        u_out_nest = ms_solver.get_opt_controls(sol) #Returns 2 lists with each input for the horizon
        x_pred_nest = ms_solver.get_pred_states(sol)

      # Get the nested list down to a single list representing the input vector
        u_out = [u_out_nest[j][i] for i in range(len(u_out_nest[0])) for j in range(len(u_out_nest))] # Merging columns instead of rows for each step in N

        #Calculate individual costs
        x_pred = [x_pred_nest[j][i] for i in range(len(x_pred_nest[0])) for j in range(len(x_pred_nest))]
        self.individual_cost(Cost_dict, problem['x'], x_pred, u_out)

        #Get the next intial guess list
        initial_guess = []
        for i in range(0, len(x_pred), self.ns):
            initial_guess.extend(x_pred[i:i+self.ns])
            if i // self.ns < len(u_out):
                initial_guess.extend(u_out[i//self.ns * self.nu : i//self.ns * self.nu + self.nu])
        initial_guess = initial_guess[5:]
        initial_guess.extend(initial_guess[-5:])

        
        return u_out, solver_cost, exit_status, solver_time, initial_guess




