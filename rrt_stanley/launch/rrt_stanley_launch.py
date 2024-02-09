from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    ld = LaunchDescription()
    config = os.path.join(
        get_package_share_directory('rrt_stanley'),
        'config',
        'config.yaml'
    )

    rrt_stanley_node = Node(
        package='rrt_stanley',
        executable='rrt_stanley',
        name='rrt_stanley',
        parameters=[config]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', os.path.join(get_package_share_directory(
            'rrt_stanley'), 'launch', 'rrt_stanley.rviz')]
    )

    waypoint_visualizer_node = Node(
        package='waypoint_visualizer',
        executable='waypoint_visualizer',
        name='waypoint_visualizer_node',
        parameters=[config]
    )

    # finalize
    ld.add_action(rviz_node)
    ld.add_action(rrt_stanley_node)
    ld.add_action(waypoint_visualizer_node)

    return ld
