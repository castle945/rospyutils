#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from rospy_msgs.cloud_publisher import CloudPublisher
from rospy_msgs.marker_publisher import MarkerPublisher
from rospy_msgs.image_publisher import ImagePublisher
from pu4c.det3d.utils import get_oriented_bounding_box_corners, get_oriented_bounding_box_lines
import numpy as np
from functools import partial
import cv2
import copy
import pandas as pd
from dataset_vis_utils import Controller, DatasetVisualizer

# -------------------------- config -------------------------- #
data_root, classes = "/datasets/KITTI/object", ['Car', 'Pedestrian', 'Cyclist']
# 1. 是否发布点云投影到图像的结果
pub_image_with_box, pub_box_attr, pub_image_with_points = True, False, False
# pub_image_with_box, pub_box_attr, pub_image_with_points = True, True, True
# 2. 边界框颜色渲染
box_color_method, box_color_param = 'label', None
# box_color_method, box_color_param = 'difficulty', None
# box_color_method, box_color_param = 'num_points_in_gt', {'level': np.array([0, 5, 100, 100000])}
# 3. 点云颜色渲染
# pc_color_method, pc_color_param = 'intensity', None
pc_color_method, pc_color_param = 'fov', {'image_shape': [370, 1240]}
# 4. 点云过滤
pc_filter_method, pc_filter_param = [], None
# pc_filter_method, pc_filter_param = ['range'], {'limit_range': [0, -40, -3, 70.4, 40, 1]}
# pc_filter_method, pc_filter_param = ['range', 'fov'], {'limit_range': [0, -40, -3, 70.4, 40, 1], 'image_shape': [370, 1240]}
# 5. 边界框过滤
# box_filter_method, box_filter_param = [], None
box_filter_method, box_filter_param = ['range'], {'limit_range': [0, -40, -3, 70.4, 40, 1]}
# 6. 输入一个或多个评估文件，命令行输入
uniform_color, eval_pkls= False, []
# classes, uniform_color, eval_pkls = ['Car'], True, []
# -------------------------- config -------------------------- #

if __name__ == '__main__':
    rospy.init_node('pcdet_vis_kitti_dataset_node', anonymous=True)

    pkl = rospy.get_param('pkl')
    start_idx = rospy.get_param('start_idx')
    eval_pkls = rospy.get_param('eval_pkls')

    infos = pd.read_pickle(pkl)
    dv = DatasetVisualizer(infos, classes[:3], default_cam='image_2')
    ctrl = Controller(idx=start_idx, len=len(infos), play=False)
    cams_keys = list(infos[0]['image'].keys())
    print(f"visualize dataset, total {len(infos)} frames")

    # 评估文件结果添加到真值的数据中，如果有评估文件则原真值框会被设置为绿色
    with_eval_info = len(eval_pkls) != 0
    for i, eval_pkl in enumerate(eval_pkls):
        infos = dv.decode_kitti_eval(infos, eval_infos=pd.read_pickle(eval_pkl), model_id=i, uniform_color=uniform_color)

    teleop_key_sub = rospy.Subscriber("/teleop_key", String, callback=partial(ctrl.onkey, step=10))
    cloud_pub = CloudPublisher("/lidar", frame_id="ego", queue_size=1, point_type="PointXYZI")
    boxes3d_pub = MarkerPublisher("/boxes3d", frame_id="ego", queue_size=1)
    cams_image_pub, cams_with_box_pub, cams_with_points_pub = {}, {}, {}
    for key in cams_keys:
        cams_image_pub[key] = ImagePublisher(key, frame_id="ego", queue_size=1)
        if pub_image_with_box:
            cams_with_box_pub[key] = ImagePublisher(f"{key}_with_box", frame_id="ego", queue_size=1)
        if pub_box_attr:
            box_attr_pub = MarkerPublisher("/box_attr", frame_id="ego", queue_size=1)
        if pub_image_with_points:
            cams_with_points_pub[key] = ImagePublisher(f"{key}_with_points", frame_id="ego", queue_size=1)

    rate = rospy.Rate(1)
    while not rospy.is_shutdown():

        idx = ctrl.idx
        info = infos[idx]
        pc = info['lidar']
        print(f"frame {idx}: {pc['filepath']}")

        points = np.fromfile(f"{data_root}/{pc['filepath']}", dtype=np.float32).reshape(-1, pc['num_features'])
        points = dv.filter_point_cloud(idx, points, method=pc_filter_method, param=pc_filter_param)
        dv.render_point_cloud_color(idx, points, method=pc_color_method, param=pc_color_param)
        cloud_pub.publish(points, rospy.Time.now())

        boxes3d = dv.transform_annos_to_boxes(info['annos'], with_eval_info=with_eval_info)
        if boxes3d.shape[0] != 0:
            boxes3d = dv.filter_boxes(idx, boxes3d, method=box_filter_method, param=box_filter_param)
            boxes3d_color = dv.render_boxes_color(idx, boxes3d, method=box_color_method, param=box_color_param)
        if boxes3d.shape[0] != 0:
            if with_eval_info:
                boxes3d, boxes3d_color = dv.merge_preds(boxes3d, boxes3d_color, info['preds'])

            corners_array = np.array([get_oriented_bounding_box_corners(b['box3d'][:3], b['box3d'][3:6], np.array([0, 0, b['box3d'][6]])) for b in boxes3d])
            lines = get_oriented_bounding_box_lines()
            boxes3d_pub.publish_boxes3d(corners_array, lines, rospy.Time.now(), 
                                            lifetime=rate.sleep_dur, colors=boxes3d_color)

        if pub_box_attr:
            boxes_text = [str(b['info']).replace(',', '\n') for b in boxes3d]
            box_attr_pub.publish_text(corners_array[:, 3], boxes_text, rospy.Time.now(),
                                      lifetime=rate.sleep_dur, scale=[0.4, 0.4, 0.4])

        for key in cams_keys:
            img = info['image'][key]
            image = cv2.imread(f"{data_root}/{img['filepath']}")
            cams_image_pub[key].publish(image, rospy.Time.now())

            if pub_image_with_box:
                image_with_boxes = dv.get_image_with_box(copy.deepcopy(image), corners_array, boxes3d_color, img['l2p_mat']) if boxes3d.shape[0] != 0 else copy.deepcopy(image)
                cams_with_box_pub[key].publish(image_with_boxes, rospy.Time.now())

            if pub_image_with_points:
                image_with_points = dv.get_image_with_points(copy.deepcopy(image), points, img['l2p_mat'])
                cams_with_points_pub[key].publish(image_with_points, rospy.Time.now())

        if ctrl.play:
            ctrl.idx = (ctrl.idx + 1) % ctrl.len
        
        rate.sleep()