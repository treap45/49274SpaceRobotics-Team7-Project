
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
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time'),
                     'detector_model': LaunchConfiguration('detector_model'),
                     'detector_input_bgr': LaunchConfiguration('detector_input_bgr')}]
    )

    ld.add_action(use_sim_time_launch_arg)
    ld.add_action(DeclareLaunchArgument(
        'detector_model',
        default_value='',
        description='YOLO weights; empty uses models/artifact_yolov8n_autolabel.pt'
    ))
    ld.add_action(DeclareLaunchArgument(
        'detector_input_bgr',
        default_value='true',
        description='false for models/artifact_yolov8n.pt (trained on channel-swapped images)'
    ))
    ld.add_action(DeclareLaunchArgument(
        'roadmap_record_path',
        default_value='',
        description='Pickle file for roadmap_replay; empty disables recording'
    ))
    ld.add_action(cave_explorer_node)
    ld.add_action(Node(
        package='cave_explorer',
        executable='cave_geometry',
        name='cave_geometry',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    ))
    ld.add_action(Node(
        package='cave_explorer',
        executable='online_roadmap',
        name='online_roadmap',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time'),
                     'record_path': LaunchConfiguration('roadmap_record_path')}]
    ))

    return ld