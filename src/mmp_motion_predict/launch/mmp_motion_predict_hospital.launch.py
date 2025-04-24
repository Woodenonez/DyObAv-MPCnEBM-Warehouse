import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node

from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.descriptions import ParameterValue


def generate_launch_description():
    ld = LaunchDescription()

    pkg_name = 'mmp_motion_predict'
    pkg_share = FindPackageShare(pkg_name).find(pkg_name) 

    ### Default Values ###
    default_timer_period = 0.2
    default_config_mmp_fname = 'hpd_1t20_poselu_enll_train.yaml'

    ### Declare Launch Variables ###
    timer_period = LaunchConfiguration('timer_period')
    config_mmp_fname = LaunchConfiguration('config_mmp_fname')

    ### Declare Launch Arguments ###
    declare_timer_period_arg = DeclareLaunchArgument(
        name="timer_period",
        default_value=str(default_timer_period),
        description="Period of timer in seconds",
    )
    ld.add_action(declare_timer_period_arg)
    
    declare_config_mmp_fname_arg = DeclareLaunchArgument(
        name="config_mmp_fname",
        default_value=default_config_mmp_fname,
        description="Config file name for motion prediction",
    )
    ld.add_action(declare_config_mmp_fname_arg)

    
    ### Nodes ###
    motion_prediction_node = Node(
        package=pkg_name,
        executable='motion_prediction_node',
        name='motion_prediction_node',
        # namespace='mmp',
        output='screen',
        parameters=[
            {'timer_period': timer_period},
            {'config_file_name': config_mmp_fname},
        ],
    )
    ld.add_action(motion_prediction_node)

    actor_traj_publisher_node = Node(
        package=pkg_name,
        executable='actor_traj_publisher',
        name='actor_traj_publisher',
        # namespace='mmp',
        output='screen',
        parameters=[
            {'timer_period': timer_period},
        ],
    )
    ld.add_action(actor_traj_publisher_node)

    return ld