from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    cfg_file = DeclareLaunchArgument('cfg_file', default_value=PathJoinSubstitution([get_package_share_directory('rospyutils'), 'config/kitti_config.yaml']), description='配置文件路径')
    pkl = DeclareLaunchArgument('pkl', default_value='src/rospyutils/data/kitti_vis_infos_val.pkl', description='')
    start_idx = DeclareLaunchArgument('start_idx', default_value='0', description='')
    run_rviz = DeclareLaunchArgument('run_rviz', default_value='true', description='Whether to run rviz')

    return LaunchDescription([
        cfg_file, pkl, start_idx, run_rviz,
        Node(
            package='rospyutils', executable='dataset_vis_node', name='dataset_vis_node', output='screen', emulate_tty=True,
            parameters=[
                {'cfg_file': LaunchConfiguration('cfg_file')},
                {'pkl': LaunchConfiguration('pkl')},
                {'start_idx': LaunchConfiguration('start_idx')},
                PathJoinSubstitution([get_package_share_directory('rospyutils'), 'config/kitti_param.yaml'])
            ]
        ),
        Node(
            condition=IfCondition(LaunchConfiguration('run_rviz')),
            package='rviz2', executable='rviz2', name='rviz2',
            arguments=['-d', PathJoinSubstitution([get_package_share_directory('rospyutils'), 'rviz/kitti_dataset_vis.rviz'])]
        ),
        ExecuteProcess(
            condition=IfCondition(LaunchConfiguration('run_rviz')),
            name='rqt_reconfigure',
            cmd=['ros2', 'run', 'rqt_reconfigure', 'rqt_reconfigure'],
        ),
    ])