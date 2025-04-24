import os
from typing import Union

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.actions import IncludeLaunchDescription

from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.substitutions import Command
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource


def summon_robot(pkg_share, robot_namespace: Union[str, LaunchConfiguration], robot_init_position: list[float]) -> list:
    """Return a list of actions (Rviz, Gazebo, MapTransform) to be added in the launch description to summon a robot."""
    launch_file = PythonLaunchDescriptionSource(
        [pkg_share, '/launch/mir_entity_spawn.launch.py'])
    include_launch = IncludeLaunchDescription(launch_file, launch_arguments={
        'namespace': robot_namespace,
        'use_gazebo': 'false',
        'use_rviz': 'false',
        'init_x': str(robot_init_position[0]),
        'init_y': str(robot_init_position[1]),
        'init_theta': str(robot_init_position[2])}.items()
    )

    if isinstance(robot_namespace, LaunchConfiguration):
        map_transform_node = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            namespace=robot_namespace,
            name='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'world', [robot_namespace, '/odom']], # x y z yaw pitch roll frame_id child_frame_id period_in_ms
        )
    elif isinstance(robot_namespace, str):
        map_transform_node = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher_'+robot_namespace,
            arguments=['0', '0', '0', '0', '0', '0', 'world', robot_namespace+'/odom'], # x y z yaw pitch roll frame_id child_frame_id period_in_ms
        )
    else:
        raise TypeError("robot_namespace must be either a string or a LaunchConfiguration.")
    
    return [include_launch, map_transform_node]


def generate_launch_description():
    ld = LaunchDescription()
  
    pkg_name = 'zmr_demo'
    pkg_share = FindPackageShare(package=pkg_name).find(pkg_name)

    ### Fixed Values ###
    robot_init_position = [-13.0, 5.85, -1.0] # x y theta

    ### Default Values ###
    default_timer_period = 0.2

    default_robot_id = '0'
    default_robot_namespace = 'mirX'

    default_map_file_name = 'zospital_map.json'
    default_graph_file_name = 'zospital_graph.json'
    default_schedule_file_name = 'zospital_schedule.csv'

    default_config_mmp_fname = 'zpd_1t20_poselu_enll_train.yaml'

    default_rviz_file_name = 'mmp_single_mir_robot_surroundings.rviz'
    default_world_file_name = 'aws/arena_hospital_small.world'
    
    ### Declare Launch Variables ###
    timer_period = LaunchConfiguration('timer_period')

    robot_id = LaunchConfiguration('robot_id')
    robot_namespace = LaunchConfiguration('robot_namespace')

    map_file_name = LaunchConfiguration('map_file_name')
    graph_file_name = LaunchConfiguration('graph_file_name')
    schedule_file_name = LaunchConfiguration('schedule_file_name')

    config_mmp_fname = LaunchConfiguration('config_mmp_fname')

    use_rviz = LaunchConfiguration('use_rviz')
    rviz_file_name = LaunchConfiguration('rviz_file_name')
    world_file_name = LaunchConfiguration('world_file_name')

    ### Declare Launch Arguments ###
    declare_timer_period_arg = DeclareLaunchArgument(
        name='timer_period',
        default_value=str(default_timer_period),
        description='Map update period'
    )
    ld.add_action(declare_timer_period_arg)

    declare_robot_id_arg = DeclareLaunchArgument(
        name='robot_id',
        default_value=str(default_robot_id),
        description='Robot ID'
    )
    ld.add_action(declare_robot_id_arg)

    declare_robot_namespace_arg = DeclareLaunchArgument(
        name='robot_namespace',
        default_value=str(default_robot_namespace),
        description='Robot namespace'
    )
    ld.add_action(declare_robot_namespace_arg)

    declare_config_mmp_fname_arg = DeclareLaunchArgument(
        name="config_mmp_fname",
        default_value=default_config_mmp_fname,
        description="Config file name for motion prediction",
    )
    ld.add_action(declare_config_mmp_fname_arg)

    declare_map_file_name_arg = DeclareLaunchArgument(
        name='map_file_name',
        default_value=str(default_map_file_name),
        description='Map file name'
    )
    ld.add_action(declare_map_file_name_arg)

    declare_graph_file_name_arg = DeclareLaunchArgument(
        name='graph_file_name',
        default_value=str(default_graph_file_name),
        description='Graph file name'
    )
    ld.add_action(declare_graph_file_name_arg)

    declare_schedule_file_name_arg = DeclareLaunchArgument(
        name='schedule_file_name',
        default_value=str(default_schedule_file_name),
        description='Schedule file name (in CSV format)'
    )
    ld.add_action(declare_schedule_file_name_arg)

    declare_use_rviz_arg = DeclareLaunchArgument(
        name='use_rviz',
        default_value='true',
        choices=["true", "false"],
        description='Whether to start RVIZ'
    )
    ld.add_action(declare_use_rviz_arg)

    declare_rviz_file_name_arg = DeclareLaunchArgument(
        name="rviz_file_name", 
        default_value=str(default_rviz_file_name), 
        description="Name of rviz config file"
    )
    ld.add_action(declare_rviz_file_name_arg)
    rviz_file_path = PathJoinSubstitution([pkg_share, 'rviz', rviz_file_name])

    declare_world_file_name_arg = DeclareLaunchArgument(
        name='world_file_name',
        default_value=str(default_world_file_name),
        description='Name of the world file to load'
    )
    ld.add_action(declare_world_file_name_arg)

    ### Nodes ###
    rviz2_node = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=["-d", rviz_file_path],
    )
    ld.add_action(rviz2_node)

    ### Other Launch Files ###
    pkg_name_1 = 'map_description'
    launch_file_1 = PythonLaunchDescriptionSource(
        [FindPackageShare(package=pkg_name_1).find(pkg_name_1), '/launch/geo_map.launch.py'])
    include_launch_1 = IncludeLaunchDescription(launch_file_1, launch_arguments={
        'timer_period': timer_period,
        'map_file_name': map_file_name,
        'use_rviz': 'false'}.items()
    )
    ld.add_action(include_launch_1)

    pkg_name_gazebo = 'gazebo_worlds'
    launch_file_gazebo = PythonLaunchDescriptionSource(
        [FindPackageShare(package=pkg_name_gazebo).find(pkg_name_gazebo), '/launch/small_warehouse.launch.py'])
    include_launch_gazebo = IncludeLaunchDescription(launch_file_gazebo, launch_arguments={
        'world_file_name': world_file_name,
        'paused': 'true'}.items()
    )
    ld.add_action(include_launch_gazebo)

    pkg_name_2 = 'mps_motion_plan'
    launch_file_2 = PythonLaunchDescriptionSource(
        [FindPackageShare(package=pkg_name_2).find(pkg_name_2), '/launch/motion_plan.launch.py'])
    include_launch_2 = IncludeLaunchDescription(launch_file_2, launch_arguments={
        "schedule_file_name": schedule_file_name,
        'graph_file_name': graph_file_name,
        'rescale_map': 'false',
        'rescale_graph': 'false'}.items()
    )
    ld.add_action(include_launch_2)

    pkg_name_3 = 'mir_description'
    pkg_share_robot = FindPackageShare(package=pkg_name_3).find(pkg_name_3)
    robot_launch_list = summon_robot(pkg_share_robot, robot_namespace, robot_init_position)
    for action in robot_launch_list:
        ld.add_action(action)

    ### Only when not using Gazebo ###
    # pkg_name_4 = 'zmr_drive_model'
    # launch_file_4 = PythonLaunchDescriptionSource(
    #     [FindPackageShare(package=pkg_name_4).find(pkg_name_4), '/launch/zmr_drive_model.launch.py'])
    # include_launch_4 = IncludeLaunchDescription(launch_file_4, launch_arguments={
    #     'robot_namespace': robot_namespace,
    #     'init_x': '1.0',
    #     'init_y': '-1.7',
    #     'init_theta': '1.57',
    #     'keep_cmd_vel': 'true'}.items()
    # )
    # ld.add_action(include_launch_4)

    pkg_name_6 = 'zmr_mpc'
    launch_file_6 = PythonLaunchDescriptionSource(
        [FindPackageShare(package=pkg_name_6).find(pkg_name_6), '/launch/zmr_mpc.launch.py'])
    include_launch_6 = IncludeLaunchDescription(launch_file_6, launch_arguments={
        'timer_period': timer_period,
        'robot_namespace': robot_namespace,
        'robot_id': robot_id}.items()
    )
    ld.add_action(include_launch_6)

    pkg_name_7 = 'mmp_motion_predict'
    launch_file_7 = PythonLaunchDescriptionSource(
        [FindPackageShare(package=pkg_name_7).find(pkg_name_7), '/launch/mmp_motion_predict.launch.py'])
    include_launch_7 = IncludeLaunchDescription(launch_file_7, launch_arguments={
        'timer_period': timer_period,
        'config_mmp_fname': config_mmp_fname}.items()
    )
    ld.add_action(include_launch_7)

    return ld