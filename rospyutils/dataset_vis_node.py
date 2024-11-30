#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import String
from rospyutils.rospy_msgs import CloudPublisher, ImagePublisher, MarkerPublisher
from rospyutils.dataset_visualizer import DatasetVisualizer, Controller
from pu4c.cv.utils import get_oriented_bounding_box_corners, get_oriented_bounding_box_lines

from functools import partial
import numpy as np
import pu4c, os
import cv2
import copy
from easydict import EasyDict
import yaml

class DatasetVisNode(Node):
    def __init__(self):
        super().__init__('dataset_vis_node')
        self.init_parameters_and_configs()

        self.teleop_key_sub = self.create_subscription(String, '/teleop_key', partial(self.ctrl.onkey, step=10), 10)
        self.cloud_pub = CloudPublisher(node=self, topic_name="/lidar", frame_id="ego", qos_profile=1, point_type="PointXYZI")
        self.bbox_3d_pub = MarkerPublisher(node=self, topic_name="/bbox_3d", frame_id="ego", qos_profile=1)
        self.cams_image_pub, self.cams_with_box_pub, self.cams_with_points_pub = {}, {}, {}
        for key in self.cfg.cams_keys:
            self.cams_image_pub[key] = ImagePublisher(node=self, topic_name=key, frame_id="ego", qos_profile=1)
            # if self.pub_image_with_box: # always create publisher for dynamic param reconfig
            self.cams_with_box_pub[key] = ImagePublisher(node=self, topic_name=f"{key}_with_box", frame_id="ego", qos_profile=1)
            self.bbox_attr_pub = MarkerPublisher(node=self, topic_name="/bbox_attr", frame_id="ego", qos_profile=1)
            self.cams_with_points_pub[key] = ImagePublisher(node=self, topic_name=f"{key}_with_points", frame_id="ego", qos_profile=1)

        self.timer = self.create_timer(1.0, self.process)
        self.add_on_set_parameters_callback(self.dynamic_param_callback)

    def init_parameters_and_configs(self):
        self.declare_parameters(
            namespace='',
            parameters=[
                # 节点命令行参数
                ('cfg_file', 'src/rospyutils/config/kitti_config.yaml'),
                ('pkl', 'src/rospyutils/data/kitti_vis_infos_val.pkl'), 
                ('start_idx', 0),
                # 动态参数，其初值由 xxx_param.yaml 文件提供，注意 yaml 文件中的第一个键名必须节点名称同步修改
                ('pub_image_with_box', False), ('pub_image_with_points', False), ('pub_bbox_attr', False),

                ('pc_color_method', 'intensity'), 
                ('box_color_method', 'None'), 
                ('pc_filter_method.range', False), ('pc_filter_method.fov', False), 
                ('bbox_filter_method.range', False), ('bbox_filter_method.fov', False), ('bbox_filter_method.num_points', False),
                
                ('scene_filter_keys.night', False), ('scene_filter_keys.rain', False), ('scene_filter_keys.reflections', False), ('scene_filter_keys.nature', False), 
            ]
        )
        # 命令行参数处理
        with open(self.get_parameter('cfg_file').value, 'r') as f:
            cfg = EasyDict(yaml.safe_load(f))

        data_infos = pu4c.read_pickle(self.get_parameter('pkl').value)
        infos = data_infos['infos']
        ctrl = Controller(idx=self.get_parameter('start_idx').value, len=len(infos))
        dv = DatasetVisualizer(infos, categories=data_infos['categories'], default_cam=cfg.default_cam)
        print(f"visualize dataset, total {len(infos)} frames")

        self.cfg, self.infos, self.ctrl, self.dv = cfg, infos, ctrl, dv
        # 保存当前的动态参数到成员变量
        self.pub_image_with_box = self.get_parameter('pub_image_with_box').value
        self.pub_image_with_points = self.get_parameter('pub_image_with_points').value
        self.pub_bbox_attr = self.get_parameter('pub_bbox_attr').value
        self.pc_color_method = self.get_parameter('pc_color_method').value
        self.box_color_method = self.get_parameter('box_color_method').value
        print(self.pub_image_with_box, self.pub_image_with_points, self.pub_bbox_attr)

        self.pc_filter_method  = [key for key in ['range', 'fov'] if self.get_parameter(f'pc_filter_method.{key}').value]
        self.bbox_filter_method = [key for key in ['range', 'fov', 'num_points'] if self.get_parameter(f'bbox_filter_method.{key}').value]
        self.scene_filter_keys = [key for key in ['night', 'rain', 'reflections', 'nature'] if self.get_parameter(f'scene_filter_keys.{key}').value]

    def dynamic_param_callback(self, params):
        # 更新当前的动态参数到成员变量
        for param in params:
            key = str(param.name).split('.')[-1]
            if param.name == 'start_idx':
                self.ctrl.idx = param.value
            elif param.name in ['pub_image_with_box', 'pub_image_with_points', 'pub_bbox_attr', 'pc_color_method', 'box_color_method']:
                setattr(self, param.name, param.value)
            elif param.name in ['pc_filter_method.range', 'pc_filter_method.fov']:
                # 正常情况下理论上只需要两个中的任意一个条件即可，因为 params 中只包含当前修改的参数，不管了先确保程序功能正常，如果出现异常 remove 会报错
                [self.pc_filter_method.append(key) if param.value and (key not in self.pc_filter_method) else self.pc_filter_method.remove(key)]
            elif param.name in ['bbox_filter_method.range', 'bbox_filter_method.fov', 'bbox_filter_method.num_points']:
                [self.bbox_filter_method.append(key) if param.value and (key not in self.bbox_filter_method) else self.bbox_filter_method.remove(key)]
            elif param.name in ['scene_filter_keys.night', 'scene_filter_keys.rain', 'scene_filter_keys.reflections', 'scene_filter_keys.nature']:
                [self.scene_filter_keys.append(key) if param.value and (key not in self.scene_filter_keys) else self.scene_filter_keys.remove(key)]
            else:
                print(f"{param.name} can not reconfig")
                return SetParametersResult(successful=False, reason=f'{param.name} can not reconfig') # 置为失败则不会修改参数的值
            
            print(f"param {param.name} changed to: {param.value}")

        return SetParametersResult(successful=True, reason='success')

    def process(self):
        # @# 帧跳转，如果动态参数给出场景描述则跳转到下一帧符合所有场景描述的帧
        if len(self.scene_filter_keys) > 0:
            # 查找一遍数据集，如果不存在不跳转
            for i in range(len(self.infos)):
                if all([(key in self.infos[self.ctrl.idx]['scene_desc']) for key in self.scene_filter_keys]):
                    break
                self.ctrl.idx = (self.ctrl.idx + 1) % self.ctrl.len

        idx = self.ctrl.idx
        info = self.infos[idx]
        lidar_info = info['lidar']
        print(f"frame {info['frame_idx']}: {lidar_info['filepath']}")
        timestamp = self.get_clock().now().to_msg()

        points = np.fromfile(os.path.join(self.cfg.data_root, lidar_info['filepath']), dtype=np.float32).reshape(-1, lidar_info['num_features'])
        points = self.dv.filter_point_cloud(idx, points, method=self.pc_filter_method, param=self.cfg.pc_filter_param)
        self.dv.render_point_cloud_color(idx, points, method=self.pc_color_method, param=self.cfg.pc_color_param)
        self.cloud_pub.publish(points, timestamp)

        bboxes_3d = np.array(info['labels']['det3d'])
        if len(bboxes_3d) != 0:
            bboxes_3d = self.dv.filter_bbox(idx, bboxes_3d, method=self.bbox_filter_method, param=self.cfg.box_filter_param)
            bboxes_3d_color = self.dv.render_bbox_color(idx, bboxes_3d, method=self.box_color_method, param=self.cfg.box_color_param)
        if len(bboxes_3d) != 0:
            corners_array = np.array([get_oriented_bounding_box_corners(b['bbox_3d'][:3], b['bbox_3d'][3:6], np.array([0, 0, b['bbox_3d'][6]])) for b in bboxes_3d])
            lines = get_oriented_bounding_box_lines()
            self.bbox_3d_pub.publish_boxes3d(corners_array, lines, timestamp, timer_period_ns=self.timer.timer_period_ns, colors=bboxes_3d_color)

        if self.pub_bbox_attr:
            attr_texts = [str(b['bbox_attr']).replace(',', '\n') for b in bboxes_3d]
            self.bbox_attr_pub.publish_text(corners_array[:, 3], attr_texts, timestamp, timer_period_ns=self.timer.timer_period_ns, scale=[0.4, 0.4, 0.4])

        for key in self.cfg.cams_keys:
            img_info = info['camera'][key]
            image = cv2.imread(os.path.join(self.cfg.data_root, img_info['filepath']))
            self.cams_image_pub[key].publish(image, timestamp)

            lidar2img = np.array(img_info['lidar2img'])
            if self.pub_image_with_box:
                image_with_boxes = self.dv.get_image_with_box(copy.deepcopy(image), corners_array, bboxes_3d_color, lidar2img) if len(bboxes_3d) != 0 else copy.deepcopy(image)
                self.cams_with_box_pub[key].publish(image_with_boxes, timestamp)

            if self.pub_image_with_points:
                image_with_points = self.dv.get_image_with_points(copy.deepcopy(image), points, lidar2img)
                self.cams_with_points_pub[key].publish(image_with_points, timestamp)

        if self.ctrl.play:
            self.ctrl.idx = (self.ctrl.idx + 1) % self.ctrl.len
        

def main(args=None):
    rclpy.init(args=args)

    node = DatasetVisNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()