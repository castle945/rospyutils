from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    cfg_file = DeclareLaunchArgument('cfg_file', default_value=PathJoinSubstitution([get_package_share_directory('rospyutils'), 'config/nuscenes_config.yaml']), description='配置文件路径')
    pkl = DeclareLaunchArgument('pkl', default_value='/workspace/files/blob/pu4c/nuscenes_vis_infos_val.pkl', description='OpenPCDet 生成的数据集预处理文件 info.pkl')
    start_idx = DeclareLaunchArgument('start_idx', default_value='0', description='')
    eval_pkls = DeclareLaunchArgument('eval_pkls', default_value='None', description='')
    run_rviz = DeclareLaunchArgument('run_rviz', default_value='true', description='Whether to run rviz')

    return LaunchDescription([
        cfg_file, pkl, start_idx, eval_pkls, run_rviz,
        Node(
            package='rospyutils', executable='pcdet_dataset_vis_node', name='pcdet_dataset_vis_node', output='screen', emulate_tty=True,
            parameters=[
                {'cfg_file': LaunchConfiguration('cfg_file')},
                {'pkl': LaunchConfiguration('pkl')},
                {'start_idx': LaunchConfiguration('start_idx')},
                {'eval_pkls': LaunchConfiguration('eval_pkls')},
                PathJoinSubstitution([get_package_share_directory('rospyutils'), 'config/nuscenes_param.yaml'])
            ]
        ),
        Node(
            condition=IfCondition(LaunchConfiguration('run_rviz')),
            package='rviz2', executable='rviz2', name='rviz2',
            arguments=['-d', PathJoinSubstitution([get_package_share_directory('rospyutils'), 'rviz/nuscenes_dataset_vis.rviz'])]
        ),
        ExecuteProcess(
            condition=IfCondition(LaunchConfiguration('run_rviz')),
            name='rqt_reconfigure',
            cmd=['ros2', 'run', 'rqt_reconfigure', 'rqt_reconfigure'],
        ),
    ])