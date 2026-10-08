
import launch_ros.actions
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import (Command, LaunchConfiguration,
                                  PathJoinSubstitution)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    ld = LaunchDescription()
    
    # Additional command line arguments
    use_sim_time_launch_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Flag to enable use_sim_time'
    )

    # Start Navigation Stack
    cave_explorer_node = Node(
        package='cave_explorer',
        executable='cave_explorer',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    ld.add_action(use_sim_time_launch_arg)
    ld.add_action(cave_explorer_node)

    return ld