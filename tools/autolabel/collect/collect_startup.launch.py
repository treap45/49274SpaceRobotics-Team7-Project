import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.substitutions import (Command, LaunchConfiguration,
                                  PathJoinSubstitution)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


HERE = os.path.dirname(os.path.abspath(__file__))


def generate_launch_description():

    ld = LaunchDescription()

    config_path = os.path.join(
        get_package_share_directory('cave_explorer'),
        'config')
    
    # Additional command line arguments
    use_sim_time_launch_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Flag to enable use_sim_time'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')
    world_launch_arg = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(HERE, 'mars_cave_labelled.sdf'),
        description='Absolute path of a world whose artefacts carry Label plugins'
    )
    world_path = LaunchConfiguration('world')
    odom_mode_launch_arg = DeclareLaunchArgument(
        'odom_mode',
        default_value='robot_localization',
        description='How to publish odom -> base_link transform',
        choices=['gazebo', 'robot_localization']
    )
    depth_pointcloud_launch_arg = DeclareLaunchArgument(
        'depth_pointcloud',
        default_value='False',
        description='Flag to 3D pointcloud from RGB-D camera'
    )

    # Load robot_description and start robot_state_publisher
    robot_description_content = ParameterValue(
        Command(['xacro ',
                 PathJoinSubstitution([
                     HERE, 'robot_labelled.urdf.xacro'])]),
        value_type=str)
    robot_state_publisher_node = Node(package='robot_state_publisher',
                                      executable='robot_state_publisher',
                                      parameters=[{
                                          'robot_description': robot_description_content,
                                          'use_sim_time': use_sim_time
                                      }])

    # Start Gazebo to simulate the robot
    gazebo = IncludeLaunchDescription(
        PathJoinSubstitution([FindPackageShare('ros_ign_gazebo'), 'launch', 'ign_gazebo.launch.py']),
        launch_arguments={
            'ign_args': [world_path, ' -r -s --headless-rendering']}.items()
    )

    # Spawn robot in Gazebo
    robot_spawner = Node(
        package='ros_ign_gazebo',
        executable='create',
        output='screen',
        arguments=['-topic', '/robot_description']
    )

    # Bridge topics between gazebo and ROS2
    gazebo_bridge = Node(
        package="ros_ign_bridge",
        executable="parameter_bridge",
        parameters=[{'config_file': os.path.join(config_path, 'gazebo_bridge_params.yaml'),
                    'use_sim_time': use_sim_time}]
    )

    # Bridge rgbd camera depth pointcloud if required
    rgbd_depth_pcl2_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='rgbd_depth_pcl2_bridge',
        output='screen',
        arguments=['/model/mars_explorer/camera/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked'],
        remappings=[('/model/mars_explorer/camera/points', '/camera/depth/points')],
        condition=IfCondition(LaunchConfiguration('depth_pointcloud'))
    )

    # Publish odom -> base_link transform **using robot_localization**
    robot_localization_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='robot_localization',
        output='screen',
        parameters=[os.path.join(config_path, 'robot_localization.yaml'),
                    {'use_sim_time': use_sim_time}],
        condition=LaunchConfigurationEquals('odom_mode', 'robot_localization')
    )

    # # Publish odom -> base_link transform **directly from gazebo**
    # odom_tf_bridge_node = Node(
    #     package='ros_gz_bridge',
    #     executable='parameter_bridge',
    #     name='odom_tf_bridge',
    #     output='screen',
    #     arguments=['/model/mars_explorer/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V'],
    #     remappings=[('/model/mars_explorer/tf', '/tf')],
    #     condition=LaunchConfigurationEquals('odom_mode', 'gazebo')
    # )

    # # Bridge gazebo odometry to either /odom or /diff_drive_controller/odom depending on odom_mode
    # odom_topic_bridge_node_robot_localization = Node(
    #     package='ros_gz_bridge',
    #     executable='parameter_bridge',
    #     name='odom_bridge',
    #     output='screen',
    #     arguments=['/model/mars_explorer/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry'],
    #     remappings=[('/model/mars_explorer/odometry', '/diff_drive_controller/odom')],
    #     condition=LaunchConfigurationEquals('odom_mode', 'robot_localization')
    # )
    # odom_topic_bridge_node_gazebo = Node(
    #     package='ros_gz_bridge',
    #     executable='parameter_bridge',
    #     name='odom_bridge',
    #     output='screen',
    #     arguments=['/model/mars_explorer/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry'],
    #     remappings=[('/model/mars_explorer/odometry', '/odom')],
    #     # condition=LaunchConfigurationEquals('odom_mode', 'gazebo')
    # )

    # rviz2 visualises data
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=['-d', os.path.join(
            config_path,
            'cave_explorer.rviz')]
    )

    ld.add_action(use_sim_time_launch_arg)
    ld.add_action(world_launch_arg)
    ld.add_action(odom_mode_launch_arg)
    ld.add_action(depth_pointcloud_launch_arg)

    ld.add_action(robot_state_publisher_node)
    # ld.add_action(odom_tf_bridge_node)
    # ld.add_action(odom_topic_bridge_node_robot_localization)
    # ld.add_action(odom_topic_bridge_node_gazebo)
    ld.add_action(gazebo)
    ld.add_action(robot_spawner)
    ld.add_action(gazebo_bridge)
    ld.add_action(rgbd_depth_pcl2_bridge_node)
    ld.add_action(Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='boxes_bridge',
        arguments=['/model/mars_explorer/boxes@vision_msgs/msg/Detection2DArray'
                   '[ignition.msgs.AnnotatedAxisAligned2DBox_V'],
        remappings=[('/model/mars_explorer/boxes', '/camera/boxes')],
        parameters=[{'use_sim_time': use_sim_time}]))
    ld.add_action(Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='segmentation_bridge',
        arguments=['/model/mars_explorer/segmentation/labels_map@sensor_msgs/msg/Image'
                   '[ignition.msgs.Image'],
        remappings=[('/model/mars_explorer/segmentation/labels_map', '/camera/labels_map')],
        parameters=[{'use_sim_time': use_sim_time}]))
    ld.add_action(robot_localization_node)
    # headless: no RViz

    return ld