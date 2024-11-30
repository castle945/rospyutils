#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rospyutils.rospy_msgs.cloud_publisher import CloudPublisher
from rospyutils.rospy_msgs.marker_publisher import MarkerPublisher
from rospyutils.rospy_msgs.image_publisher import ImagePublisher
from pu4c.det3d.utils import get_oriented_bounding_box_corners, get_oriented_bounding_box_lines
import numpy as np
from functools import partial
import cv2
import copy
import pandas as pd
from rospyutils.dataset_vis_utils import ctrl, DatasetVisualizer

# -------------------------- config -------------------------- #
data_root, classes = "/datasets/KITTI/object", ['Car', 'Pedestrian', 'Cyclist']
# 1. 是否发布点云投影到图像的结果
# pub_image_with_box, pub_box_attr, pub_image_with_points = True, False, False
pub_image_with_box, pub_box_attr, pub_image_with_points = True, True, True
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

class VisKittiNode(Node):
    def __init__(self):
        super().__init__('pcdet_vis_kitti_dataset_node')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('pkl', '/workspace/files/blob/pu4c/kitti_vis_infos_val.pkl'), 
                ('eval_pkls', '[]'), 
                ('start_idx', 0),
            ]
        )
        info_pkl = self.get_parameter('pkl').get_parameter_value().string_value
        start_idx = self.get_parameter('start_idx').get_parameter_value().integer_value
        eval_pkls = [] if self.get_parameter('eval_pkls').get_parameter_value().string_value == 'None' else eval(self.get_parameter('eval_pkls').get_parameter_value().string_value) # 字符串转代码，即将字符串列表'[]'转列表，launch 文件甚至不支持传入 '[]'

        infos = pd.read_pickle(info_pkl)
        dv = DatasetVisualizer(infos, classes[:3], default_cam='image_2')
        ctrl.idx, ctrl.len = start_idx, len(infos)
        cams_keys = list(infos[0]['image'].keys())
        print(f"visualize dataset, total {len(infos)} frames")
        # 评估文件结果添加到真值的数据中，如果有评估文件则原真值框会被设置为绿色
        for i, eval_pkl in enumerate(eval_pkls):
            infos = dv.decode_kitti_eval(infos, eval_infos=pd.read_pickle(eval_pkl), model_id=i, uniform_color=uniform_color)
        self.infos, self.dv, self.cams_keys, self.eval_pkls = infos, dv, cams_keys, eval_pkls

        self.teleop_key_sub = self.create_subscription(String, '/teleop_key', partial(ctrl.onkey, step=10), 10)
        self.cloud_pub = CloudPublisher(node=self, topic_name="/lidar", frame_id="ego", qos_profile=1, point_type="PointXYZI")
        self.boxes3d_pub = MarkerPublisher(node=self, topic_name="/boxes3d", frame_id="ego", qos_profile=1)
        self.cams_image_pub, self.cams_with_box_pub, self.cams_with_points_pub = {}, {}, {}
        for key in cams_keys:
            self.cams_image_pub[key] = ImagePublisher(node=self, topic_name=key, frame_id="ego", qos_profile=1)
            if pub_image_with_box:
                self.cams_with_box_pub[key] = ImagePublisher(node=self, topic_name=f"{key}_with_box", frame_id="ego", qos_profile=1)
            if pub_box_attr:
                self.box_attr_pub = MarkerPublisher(node=self, topic_name="/box_attr", frame_id="ego", qos_profile=1)
            if pub_image_with_points:
                self.cams_with_points_pub[key] = ImagePublisher(node=self, topic_name=f"{key}_with_points", frame_id="ego", qos_profile=1)

        self.timer = self.create_timer(1.0, self.process)

    def process(self):
        idx = ctrl.idx
        info = self.infos[idx]
        pc = info['lidar']
        print(f"frame {idx}: {pc['filepath']}")
        timestamp = self.get_clock().now().to_msg()

        points = np.fromfile(f"{data_root}/{pc['filepath']}", dtype=np.float32).reshape(-1, pc['num_features'])
        points = self.dv.filter_point_cloud(idx, points, method=pc_filter_method, param=pc_filter_param)
        self.dv.render_point_cloud_color(idx, points, method=pc_color_method, param=pc_color_param)
        self.cloud_pub.publish(points, timestamp)

        with_eval_info = len(self.eval_pkls) != 0
        boxes3d = self.dv.transform_annos_to_boxes(info['annos'], with_eval_info=with_eval_info)
        if boxes3d.shape[0] != 0:
            boxes3d = self.dv.filter_boxes(idx, boxes3d, method=box_filter_method, param=box_filter_param)
            boxes3d_color = self.dv.render_boxes_color(idx, boxes3d, method=box_color_method, param=box_color_param)
        if boxes3d.shape[0] != 0:
            if with_eval_info:
                boxes3d, boxes3d_color = self.dv.merge_preds(boxes3d, boxes3d_color, info['preds'])

            corners_array = np.array([get_oriented_bounding_box_corners(b['box3d'][:3], b['box3d'][3:6], np.array([0, 0, b['box3d'][6]])) for b in boxes3d])
            lines = get_oriented_bounding_box_lines()
            self.boxes3d_pub.publish_boxes3d(corners_array, lines, timestamp, timer_period_ns=self.timer.timer_period_ns, colors=boxes3d_color)

        if pub_box_attr:
            boxes_text = [str(b['info']).replace(',', '\n') for b in boxes3d]
            self.box_attr_pub.publish_text(corners_array[:, 3], boxes_text, timestamp, timer_period_ns=self.timer.timer_period_ns, scale=[0.4, 0.4, 0.4])

        for key in self.cams_keys:
            img = info['image'][key]
            image = cv2.imread(f"{data_root}/{img['filepath']}")
            self.cams_image_pub[key].publish(image, timestamp)

            if pub_image_with_box:
                image_with_boxes = self.dv.get_image_with_box(copy.deepcopy(image), corners_array, boxes3d_color, img['l2p_mat']) if boxes3d.shape[0] != 0 else copy.deepcopy(image)
                self.cams_with_box_pub[key].publish(image_with_boxes, timestamp)

            if pub_image_with_points:
                image_with_points = self.dv.get_image_with_points(copy.deepcopy(image), points, img['l2p_mat'])
                self.cams_with_points_pub[key].publish(image_with_points, timestamp)

        if ctrl.play:
            ctrl.idx = (ctrl.idx + 1) % ctrl.len
        

def main(args=None):
    rclpy.init(args=args)

    node = VisKittiNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()