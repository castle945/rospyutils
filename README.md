* 代码运行

  * ```bash
    roslaunch rospyutils vis_kitti_dataset.launch rviz:=True pkl:=/workspace/files/blob/pu4c/kitti_vis_infos_val.pkl
    rosrun rospyutils teleop_key_node.py

    roslaunch rospyutils vis_nus_dataset.launch rviz:=True pkl:=/workspace/files/blob/pu4c/nuscenes_vis_infos_val.pkl
    roslaunch rospyutils vis_lidaronly_dataset.launch rviz:=True pkl:=/workspace/files/blob/Det3DTrans/OpenPCDetTrans/data/lidarcs/VLD-64/lidarcs_infos_val.pkl
    roslaunch rospyutils vis_kitti_dataset.launch rviz:=True eval_pkls:=["/datasets/blob/ST3D/model_zoo/20240116-163105/eval/eval_with_train/epoch_40/val/result.pkl","/datasets/blob/ST3D/model_zoo/20240117-024819/eval/eval_with_train/epoch_12/val/result.pkl"]
    ```