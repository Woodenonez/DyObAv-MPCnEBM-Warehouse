import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, ThisLaunchFileDir, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    ld = LaunchDescription()

    pkg_name = 'gazebo_worlds'
    pkg_share = FindPackageShare(package=pkg_name).find(pkg_name)

    # Define launch arguments
    declare_world_file_name_arg = DeclareLaunchArgument(
        name='world_file_name',
        default_value='aws/hospital.world',
        description='Relative path to the world model file to load'
    )
    ld.add_action(declare_world_file_name_arg)

    declare_paused_arg = DeclareLaunchArgument(
        'paused', default_value='true',
        description='Start the simulation in a paused state'
    )
    ld.add_action(declare_paused_arg)

    # declare_use_sim_time_arg = DeclareLaunchArgument(
    #     'use_sim_time', default_value='true',
    #     description='Use simulation (Gazebo) clock if true'
    # )
    # ld.add_action(declare_use_sim_time_arg)

    # declare_gui_arg = DeclareLaunchArgument(
    #     'gui', default_value='true',
    #     description='Launch Gazebo GUI'
    # )
    # ld.add_action(declare_gui_arg)

    # declare_headless_arg = DeclareLaunchArgument(
    #     'headless', default_value='false',
    #     description='Launch Gazebo without GUI (headless mode)'
    # )
    # ld.add_action(declare_headless_arg)

    # declare_debug_arg = DeclareLaunchArgument(
    #     'debug', default_value='false',
    #     description='Start Gazebo in debug mode'
    # )
    # ld.add_action(declare_debug_arg)

    # declare_verbose_arg = DeclareLaunchArgument(
    #     'verbose', default_value='true',
    #     description='Enable verbose Gazebo output'
    # )
    # ld.add_action(declare_verbose_arg)

    # declare_recording_arg = DeclareLaunchArgument(
    #     'recording', default_value='false',
    #     description='Enable Gazebo state log recording'
    # )
    # ld.add_action(declare_recording_arg)

    # Include Gazebo launch
    world_file_path = PathJoinSubstitution([pkg_share, 'worlds', LaunchConfiguration('world_file_name')])
    gazebo = ExecuteProcess(
        cmd=[
            'gazebo', 
            '--verbose', 
            world_file_path, 
            '-s', 'libgazebo_ros_init.so', 
            '-s', 'libgazebo_ros_factory.so',
            '--pause', LaunchConfiguration('paused'),
        ],
        output='screen')
    ld.add_action(gazebo)

    return ld
