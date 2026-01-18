# Dynamic Obstacle Avoidance: One-shot Prediction (EBM) and Control (MPC)
This repository contains the ROS 2 code for the corresponding [Python counterpart](https://github.com/Woodenonez/DyObAv-MPCnEBM-Warehouse).
Please find more descriptions and related publications there.

The map generation can be found [here](https://github.com/arshadlab/gazebo_map_creator).

Before start:
- This code is tested under [ROS 2 Humble](https://docs.ros.org/en/humble/index.html).
- The warehouse Gazebo world is based on [AWS Small Warehouse](https://github.com/aws-robotics/aws-robomaker-small-warehouse-world).
- The MiR mobile robot modeling is based on [MiR Robot](https://github.com/DFKI-NI/mir_robot).
Please follow the attached licenses and contact me if there is any problem with them.

## Quick start
### MPC build
To run any MPC code, you need to build the MPC package first.
```
python [project_dir]/src/zmr_mpc/build_solver.py
``` 
It is crucial to select the right configuration for the solver. 
```
cfg_fname = "mpc_fast.yaml" # or other MPC configurations
robot_spec = "robot_spec_mir.yaml" # or other robot specs
```

### Pseudo Pure pursuit controller (Rviz - Single robot)
Launch everything in the simulation environment.

```
ros2 launch zmr_demo single_robot_pp_rviz.launch.py
```

### Pseudo Pure pursuit controller (Gazebo - Dual robots)
First, launch everything in the simulation environment.

```
ros2 launch zmr_demo dual_robot_pp_gazebo.launch.py use_controller:=false
```

Then, run the pure pursuit controller.

```
ros2 launch zmr_demo dual_robot_pp.launch.py
```

If in the first step, the `use_controller` is set to `true` (default), the pure pursuit controller will be launched automatically. For better monitoring, it is recommended to launch the controller separately.

### Model Predictive Control (MPC) (Gazebo - Dual robots)
First, launch the robot manager for fleet coordination.
```
ros2 launch zmr_manager zmr_manager.launch.py
```
Then, launch the demo.
```
ros2 launch zmr_demo dual_robot_mpc_gazebo.launch.py enable_fleet_manager:=true
```
To compare with non-coordinated robots, set `enable_fleet_manager` to `false`.

### Model Predictive Control + Energy-based Motion Prediction (MPC+EBM) (Gazebo - Warehouse Single robots)
Launch the demo with a MiR robot
```
ros2 launch zmr_demo demo_warehouse_single_robot_gazebo_mpc_mmp.launch.py
```

### Model Predictive Control + Energy-based Motion Prediction (MPC+EBM) (Gazebo - Warehouse Dual robots)
First, launch the robot manager for fleet coordination.
```
ros2 launch zmr_manager zmr_manager.launch.py robot_prefix:=mir
```
Then, launch the demo.
```
ros2 launch zmr_demo demo_mir_warehouse_dual_robot_gazebo_mpc_mmp.launch.py enable_fleet_manager:=true
```
