import pu4c, rich

def check_kitti_pkls():
    # python3 tools/create_vis_infos.py --dataset kitti --codebase pcdet --data_root /workspace/codevault/Det3D/OpenPCDet/data/kitti
    # mv data/kitti_vis_infos_val.pkl data/pcdet_kitti_vis_infos_val.pkl
    # python3 tools/create_vis_infos.py --dataset kitti --codebase mmdet3d --data_root /workspace/codevault/Det3D/mmdetection3d/data/kitti
    infos1 = pu4c.read_pickle('data/pcdet_kitti_vis_infos_val.pkl')
    infos2 = pu4c.read_pickle('data/kitti_vis_infos_val.pkl')
    assert pu4c.deep_equal(infos1, infos2, tol=(1.e-4, 1.e-5), ignore_keys=['difficulty', 'num_points']) # 可能由于计算差异有些不相等，忽略这些不重要

def check_nus_pkls():
    # python3 tools/create_vis_infos.py --dataset nuscenes --codebase pcdet --data_root /workspace/codevault/Det3D/OpenPCDet/data/nuscenes/v1.0-trainval
    # mv data/nuscenes_vis_infos_val.pkl data/pcdet_nuscenes_vis_infos_val.pkl
    # python3 tools/create_vis_infos.py --dataset nuscenes --codebase mmdet3d --data_root /workspace/codevault/Det3D/mmdetection3d/data/nuscenes
    infos1 = pu4c.read_pickle('data/pcdet_nuscenes_vis_infos_val.pkl')
    infos2 = pu4c.read_pickle('data/nuscenes_vis_infos_val.pkl')
    # 相对来说浮点数误差还蛮大的但能确定是一个值，而且 bbox_3d 中存在角度相差 2pi 的情况，即使 limit_period 到 2pi 也消不掉，不过不影响是同一个意思
    assert pu4c.deep_equal(infos1, infos2, tol=(1.e-2, 1.e-5), ignore_keys=['num_points', 'bbox_3d', 'dynamic'])

if __name__ == '__main__':
    check_nus_pkls()