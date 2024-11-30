# 适配器，将常见代码库生成的数据集信息文件统一为自定义的可视化信息格式
import argparse
import numpy as np
import pu4c, os

def box_camera_to_lidar(data, lidar2cam):
    """copy from mmdet3d/structures/ops/box_np_ops.py"""
    def limit_period(val: np.ndarray, offset: float = 0.5, period: float = np.pi) -> np.ndarray:
        limited_val = val - np.floor(val / period + offset) * period
        return limited_val
    def camera_to_lidar(points, lidar2cam):
        points_shape = list(points.shape[0:-1])
        if points.shape[-1] == 3:
            points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
        lidar_points = points @ np.linalg.inv(lidar2cam.T)
        return lidar_points[..., :3]

    xyz = data[:, 0:3]
    x_size, y_size, z_size = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    r = data[:, 6:7]
    xyz_lidar = camera_to_lidar(xyz, lidar2cam)
    # yaw and dims also needs to be converted
    r_new = -r - np.pi / 2
    # r_new = limit_period(r_new, period=np.pi * 2) # 统一一下，OpenPCDet 数据集信息中暂不限制角度周期而是在数据增强之后做的
    return np.concatenate([xyz_lidar, x_size, z_size, y_size, r_new], axis=1)

kitti_categories = {'Pedestrian': 0, 'Cyclist': 1, 'Car': 2, 'Van': 3, 'Truck': 4, 'Person_sitting': 5, 'Tram': 6, 'Misc': 7}
def create_kitti_vis_infos_from_mmdet3d(data_root, out_dir):
    for split in ['train', 'val']:
        input_file = os.path.join(data_root, f'kitti_infos_{split}.pkl')
        output_file = os.path.join(out_dir, f'kitti_vis_infos_{split}.pkl')
        if os.path.exists(output_file):
            print(f'Warning, you are overwriting the file of {output_file}.')
        data = pu4c.read_pickle(input_file)
        data['metainfo']['categories'].pop('DontCare')
        assert kitti_categories == data['metainfo']['categories']
        infos = data['data_list']
        
        split_key = 'training' if (split in ['train', 'val']) else 'testing'
        vis_infos = []
        for info in infos:
            valid_objs = [obj for obj in info['instances'] if obj['bbox_label'] != -1] # 去除 DontCare 目标后的有效目标
            # 相机坐标系下三维框转激光雷达坐标系下三维框
            bbox_3d_camera = np.array([obj['bbox_3d'] for obj in valid_objs])
            # assert np.equal(np.round(np.array(info['images']['R0_rect']) @ np.array(info['lidar_points']['Tr_velo_to_cam']), 5), np.round(np.array(info['images']['CAM2']['lidar2cam']), 5)).all()
            bbox_3d_lidar = box_camera_to_lidar(bbox_3d_camera, np.array(info['images']['CAM2']['lidar2cam'])) # lidar2cam = R0_rect @ Tr_velo_to_cam
            bbox_3d_lidar[:, 2] += (bbox_3d_lidar[:, 5] / 2)
            bbox_3d_lidar = bbox_3d_lidar.tolist()
            vis_infos.append({
                'frame_idx': str(info['sample_idx']).zfill(6), 
                'scene_desc': '', # 场景描述
                'lidar': {
                    'filepath': os.path.join(split_key, 'velodyne', info['lidar_points']['lidar_path']),
                    'num_features': 4,
                }, 
                'camera': {
                    'CAM2': {
                        'filepath': os.path.join(split_key, 'image_2', info['images']['CAM2']['img_path']),
                        'lidar2img': info['images']['CAM2']['lidar2img'],
                    },
                }, 
                'labels': {
                    'det3d': [
                        {
                            'bbox_3d': bbox_3d_lidar[i],
                            'bbox_label': obj['bbox_label'],
                            'bbox_attr': {
                                'truncated': obj['truncated'],
                                'occluded': obj['occluded'], 
                                'difficulty': obj['difficulty'], 
                                'num_points': obj['num_lidar_pts'], 
                            }
                        }
                        for i, obj in enumerate(valid_objs)
                    ],
                },
            })
    
        pu4c.write_pickle(output_file, {'categories': kitti_categories, 'infos': vis_infos})
        print(f'create {output_file}')
def create_kitti_vis_infos_from_pcdet(data_root, out_dir):
    for split in ['train', 'val']:
        input_file = os.path.join(data_root, f'kitti_infos_{split}.pkl')
        output_file = os.path.join(out_dir, f'kitti_vis_infos_{split}.pkl')
        if os.path.exists(output_file):
            print(f'Warning, you are overwriting the file of {output_file}.')
        infos = pu4c.read_pickle(input_file)
        
        split_key = 'training' if (split in ['train', 'val']) else 'testing'
        vis_infos = []
        for info in infos:
            lidar2img = info['calib']['P2'] @ info['calib']['R0_rect'] @ info['calib']['Tr_velo_to_cam']
            num_objects = info['annos']['gt_boxes_lidar'].shape[0] # 去除 DontCare 目标后的有效目标个数
            for key in ['gt_boxes_lidar', 'name', 'truncated', 'occluded', 'difficulty', 'num_points_in_gt']:
                info['annos'][key] = info['annos'][key].tolist()
            vis_infos.append({
                'frame_idx': info['point_cloud']['lidar_idx'], 
                'scene_desc': '', # 场景描述
                'lidar': {
                    'filepath': os.path.join(split_key, 'velodyne', f"{info['point_cloud']['lidar_idx']}.bin"),
                    'num_features': 4,
                },
                'camera': {
                    'CAM2': {
                        'filepath': os.path.join(split_key, 'image_2', f"{info['image']['image_idx']}.png"),
                        'lidar2img': lidar2img.tolist(),
                    },
                },
                'labels': {
                    'det3d': [
                        {
                            'bbox_3d': info['annos']['gt_boxes_lidar'][i],
                            'bbox_label': kitti_categories[info['annos']['name'][i]],
                            'bbox_attr': {
                                'truncated': info['annos']['truncated'][i],
                                'occluded': int(info['annos']['occluded'][i]),
                                'difficulty': info['annos']['difficulty'][i],
                                'num_points': info['annos']['num_points_in_gt'][i],
                            }
                        }
                        for i in range(num_objects)
                    ],
                },
            })
    
        pu4c.write_pickle(output_file, {'categories': kitti_categories, 'infos': vis_infos})
        print(f'create {output_file}')

nus_categories = {'car': 0, 'truck': 1, 'trailer': 2, 'bus': 3, 'construction_vehicle': 4, 'bicycle': 5, 'motorcycle': 6, 'pedestrian': 7, 'traffic_cone': 8, 'barrier': 9}
def create_nus_vis_infos_from_mmdet3d(data_root, out_dir):
    for split in ['train', 'val']:
        input_file = os.path.join(data_root, f'nuscenes_infos_{split}.pkl')
        output_file = os.path.join(out_dir, f'nuscenes_vis_infos_{split}.pkl')
        if os.path.exists(output_file):
            print(f'Warning, you are overwriting the file of {output_file}.')
        print(f'loading original infos from {input_file}...')
        data = pu4c.read_pickle(input_file)
        assert nus_categories == data['metainfo']['categories']
        infos = data['data_list']
        
        vis_infos = []
        for info in infos:
            # 计算并添加 lidar2img
            for cam_type, cam_info in info['images'].items():
                lidar2cam = np.array(cam_info['lidar2cam'])
                cam2img = np.eye(4)
                cam2img[:3, :3] = np.array(cam_info['cam2img']) 
                lidar2img = cam2img @ lidar2cam
                cam_info['lidar2img'] = lidar2img.tolist()
            vis_infos.append({
                'frame_idx': info['token'], 
                'scene_desc': '', # 场景描述
                'lidar': {
                    'filepath': os.path.join('samples/LIDAR_TOP', info['lidar_points']['lidar_path']),
                    'num_features': 5, # xyzi,ring
                }, 
                'camera': {
                    cam_type: {
                        'filepath': os.path.join('samples', cam_type, cam_info['img_path']),
                        'lidar2img': cam_info['lidar2img'],
                    }
                    for cam_type, cam_info in info['images'].items()
                }, 
                'labels': {
                    'det3d': [
                        {
                            'bbox_3d': obj['bbox_3d'],
                            'bbox_label': obj['bbox_label_3d'],
                            'bbox_attr': {
                                'num_points': int(obj['num_lidar_pts']), 
                                'dynamic': bool(obj['velocity'][0] > 1e-2 or obj['velocity'][1] > 1e-2),
                            }
                        }
                        for obj in info['instances'] if (obj['bbox_3d_isvalid'] and (obj['bbox_label_3d'] != -1))
                    ],
                },
            })
    
        if True:
            version = 'v1.0-mini' if 'v1.0-mini' in data_root else 'v1.0-trainval'
            nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
            # 添加场景描述信息用于可视化时，过滤样本
            for info in vis_infos:
                token = info['frame_idx']
                sample = nusc.get('sample', token)
                scene = nusc.get('scene', sample['scene_token'])
                info['scene_desc'] = scene['description'].lower()
        
        pu4c.write_pickle(output_file, {'categories': nus_categories, 'infos': vis_infos})
        print(f'create {output_file}')
def create_nus_vis_infos_from_pcdet(data_root, out_dir, add_ext_info=False):
    for split in ['train', 'val']:
        input_file = os.path.join(data_root, f'nuscenes_infos_10sweeps_{split}.pkl') # data_root 需要精确到 v1.0-trainval/v1.0-mini
        output_file = os.path.join(out_dir, f'nuscenes_vis_infos_{split}.pkl')
        if os.path.exists(output_file):
            print(f'Warning, you are overwriting the file of {output_file}.')
        print(f'loading original infos from {input_file}...')
        infos = pu4c.read_pickle(input_file)

        vis_infos = []
        for info in infos:
            ego2lidar = info['ref_from_car']
            global2ego_lidar = info['car_from_global']
            lidar2ego = transform_matrix(rotation=Quaternion._from_matrix(ego2lidar[:3, :3]), translation=ego2lidar[:3, 3], inverse=True)
            ego2global_lidar = transform_matrix(rotation=Quaternion._from_matrix(global2ego_lidar[:3, :3]), translation=global2ego_lidar[:3, 3], inverse=True)
            cam_infos = {}
            for cam_type, cam_info in info['cams'].items():
                cam2img = np.eye(4)
                cam2img[:3, :3] = cam_info['camera_intrinsics']
                global2ego_cam = transform_matrix(rotation=Quaternion(cam_info['ego2global_rotation']), translation=np.array(cam_info['ego2global_translation']), inverse=True)
                ego2cam = transform_matrix(rotation=Quaternion(cam_info['sensor2ego_rotation']), translation=np.array(cam_info['sensor2ego_translation']), inverse=True)

                lidar2img = cam2img @ ego2cam @ global2ego_cam @ ego2global_lidar @ lidar2ego
                cam_infos[cam_type] = {
                    'filepath': cam_info['data_path'],
                    'lidar2img': lidar2img.tolist(),
                }
            mask = info['gt_names'] != 'ignore'
            for key in ['gt_boxes', 'gt_names', 'num_lidar_pts']:
                info[key] = info[key][mask].tolist()
            vis_infos.append({
                'frame_idx': info['token'], 
                'lidar': {
                    'filepath': info['lidar_path'],
                    'num_features': 5,
                }, 
                'camera': cam_infos, 
                'labels': {
                    'det3d': [
                        {
                            'bbox_3d': info['gt_boxes'][i][:7],
                            'bbox_label': nus_categories[name],
                            'bbox_attr': {
                                'num_points': info['num_lidar_pts'][i],
                                'dynamic': bool(info['gt_boxes'][i][7] > 1e-2 or info['gt_boxes'][i][8] > 1e-2),
                            }
                        }
                        for i, name in enumerate(info['gt_names'])
                    ]
                }
            })

        if True:
            version = 'v1.0-mini' if 'v1.0-mini' in data_root else 'v1.0-trainval'
            nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
            # 添加场景描述信息用于可视化时，过滤样本
            for info in vis_infos:
                token = info['frame_idx']
                sample = nusc.get('sample', token)
                scene = nusc.get('scene', sample['scene_token'])
                info['scene_desc'] = scene['description'].lower()

        pu4c.write_pickle(output_file, {'categories': nus_categories, 'infos': vis_infos})
        print(f'create {output_file}')
def parse_arg():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataset',  type=str, default='kitti',   help='')
    parser.add_argument('--codebase', type=str, default='mmdet3d', help='')
    parser.add_argument('--data_root',type=str, default='/workspace/codevault/Det3D/mmdetection3d/data/kitti', help='')
    parser.add_argument('--out_dir',  type=str, default='data/',   help='')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arg()
    if args.dataset == 'kitti': 
        assert args.codebase in ['mmdet3d', 'pcdet']
        if args.codebase == 'mmdet3d':
            create_kitti_vis_infos_from_mmdet3d(data_root=args.data_root, out_dir=args.out_dir)
        elif args.codebase == 'pcdet':
            create_kitti_vis_infos_from_pcdet(data_root=args.data_root, out_dir=args.out_dir)
    elif args.dataset == 'nuscenes': 
        assert args.codebase in ['mmdet3d', 'pcdet']
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion
        if args.codebase == 'mmdet3d':
            create_nus_vis_infos_from_mmdet3d(data_root=args.data_root, out_dir=args.out_dir)
        elif args.codebase == 'pcdet':
            create_nus_vis_infos_from_pcdet(data_root=args.data_root, out_dir=args.out_dir)
    else:
        raise NotImplementedError
