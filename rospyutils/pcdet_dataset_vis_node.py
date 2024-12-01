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
from easydict import EasyDict
import yaml
import ast
from rospyutils.dataset_vis_utils import Controller, DatasetVisualizer
from rcl_interfaces.msg import SetParametersResult

class OpenPCDetDatasetVisNode(Node):
    def __init__(self):
        super().__init__('pcdet_dataset_vis_node')
        self.init_parameters_and_configs()

        self.teleop_key_sub = self.create_subscription(String, '/teleop_key', partial(self.ctrl.onkey, step=10), 10)
        self.cloud_pub = CloudPublisher(node=self, topic_name="/lidar", frame_id="ego", qos_profile=1, point_type="PointXYZI")
        self.boxes3d_pub = MarkerPublisher(node=self, topic_name="/boxes3d", frame_id="ego", qos_profile=1)
        self.cams_image_pub, self.cams_with_box_pub, self.cams_with_points_pub = {}, {}, {}
        for key in self.cfg.cams_keys:
            self.cams_image_pub[key] = ImagePublisher(node=self, topic_name=key, frame_id="ego", qos_profile=1)
            # if self.pub_image_with_box: # always create publisher for dynamic param reconfig
            self.cams_with_box_pub[key] = ImagePublisher(node=self, topic_name=f"{key}_with_box", frame_id="ego", qos_profile=1)
            self.box_attr_pub = MarkerPublisher(node=self, topic_name="/box_attr", frame_id="ego", qos_profile=1)
            self.cams_with_points_pub[key] = ImagePublisher(node=self, topic_name=f"{key}_with_points", frame_id="ego", qos_profile=1)

        self.timer = self.create_timer(1.0, self.process)
        self.add_on_set_parameters_callback(self.dynamic_param_callback)

    def init_parameters_and_configs(self):
        self.declare_parameters(
            namespace='',
            parameters=[
                # 节点命令行参数
                ('cfg_file', 'src/rospyutils/config/kitti_config.yaml'),
                ('pkl', '/workspace/files/blob/pu4c/kitti_vis_infos_val.pkl'), 
                ('eval_pkls', '[]'), 
                ('start_idx', 0),
                # 配置文件参数
                ('pub_image_with_box', False), ('pub_image_with_points', False), ('pub_box_attr', False),

                ('pc_color_method', 'intensity'), 
                ('box_color_method', 'None'), 
                ('pc_filter_method.range', False), ('pc_filter_method.fov', False), 
                ('box_filter_method.range', False), ('box_filter_method.fov', False), ('box_filter_method.num_points', False),
                
                ('scene_filter_keys.night', False), ('scene_filter_keys.rain', False), ('scene_filter_keys.reflections', False), ('scene_filter_keys.nature', False), 
            ]
        )
        # 命令行参数处理
        with open(self.get_parameter('cfg_file').value, 'r') as f:
            cfg = EasyDict(yaml.safe_load(f))

        infos = pd.read_pickle(self.get_parameter('pkl').value)
        ctrl = Controller(idx=self.get_parameter('start_idx').value, len=len(infos))
        dv = DatasetVisualizer(infos, cfg.classes, default_cam=cfg.default_cam)
        print(f"visualize dataset, total {len(infos)} frames")

        # 预测结果合并真值中，如果有预测结果则真值框会被设置为绿色
        eval_pkls = [] if self.get_parameter('eval_pkls').value == 'None' else ast.literal_eval(self.get_parameter('eval_pkls').value)
        for i, eval_pkl in enumerate(eval_pkls):
            infos = dv.decode_kitti_eval(infos, eval_infos=pd.read_pickle(eval_pkl), model_id=i)
        
        self.cfg, self.infos, self.ctrl, self.dv, self.with_eval_info = cfg, infos, ctrl, dv, len(eval_pkls) != 0
        # 配置文件参数处理
        self.pub_image_with_box = self.get_parameter('pub_image_with_box').value
        self.pub_image_with_points = self.get_parameter('pub_image_with_points').value
        self.pub_box_attr = self.get_parameter('pub_box_attr').value
        self.pc_color_method = self.get_parameter('pc_color_method').value
        self.box_color_method = self.get_parameter('box_color_method').value

        self.pc_filter_method  = [key for key in ['range', 'fov'] if self.get_parameter(f'pc_filter_method.{key}').value]
        self.box_filter_method = [key for key in ['range', 'fov', 'num_points'] if self.get_parameter(f'box_filter_method.{key}').value]
        self.scene_filter_keys = [key for key in ['night', 'rain', 'reflections', 'nature'] if self.get_parameter(f'scene_filter_keys.{key}').value]

    def dynamic_param_callback(self, params):
        for param in params:
            key = str(param.name).split('.')[-1]
            if param.name == 'start_idx':
                self.ctrl.idx = param.value
            elif param.name in ['pub_image_with_box', 'pub_image_with_points', 'pub_box_attr', 'pc_color_method', 'box_color_method']:
                setattr(self, param.name, param.value)
            elif param.name in ['pc_filter_method.range', 'pc_filter_method.fov']:
                # 正常情况下理论上只需要两个中的任意一个条件即可，因为 params 中只包含当前修改的参数，不管了先确保程序功能正常，如果出现异常 remove 会报错
                [self.pc_filter_method.append(key) if param.value and (key not in self.pc_filter_method) else self.pc_filter_method.remove(key)]
            elif param.name in ['box_filter_method.range', 'box_filter_method.fov', 'box_filter_method.num_points']:
                [self.box_filter_method.append(key) if param.value and (key not in self.box_filter_method) else self.box_filter_method.remove(key)]
            elif param.name in ['scene_filter_keys.night', 'scene_filter_keys.rain', 'scene_filter_keys.reflections', 'scene_filter_keys.nature']:
                [self.scene_filter_keys.append(key) if param.value and (key not in self.scene_filter_keys) else self.scene_filter_keys.remove(key)]
            else:
                print(f"{param.name} can not reconfig")
                return SetParametersResult(successful=False, reason=f'{param.name} can not reconfig') # 置为失败则不会修改参数的值
            
            print(f"param {param.name} changed to: {param.value}")

        return SetParametersResult(successful=True, reason='success')

    def process(self):
        # 如果有场景描述则跳转到下一帧符合所有场景描述的点云
        if len(self.scene_filter_keys) > 0:
            for i in range(len(self.infos)):
                mask = [(key in self.infos[self.ctrl.idx]['lidar']['desc']) for key in self.scene_filter_keys]
                if all(mask):
                    break
                self.ctrl.idx = (self.ctrl.idx + 1) % self.ctrl.len

        idx = self.ctrl.idx
        info = self.infos[idx]
        pc = info['lidar']
        print(f"frame {idx}: {pc['filepath']}")
        timestamp = self.get_clock().now().to_msg()

        points = np.fromfile(f"{self.cfg.data_root}/{pc['filepath']}", dtype=np.float32).reshape(-1, pc['num_features'])
        points = self.dv.filter_point_cloud(idx, points, method=self.pc_filter_method, param=self.cfg.pc_filter_param)
        self.dv.render_point_cloud_color(idx, points, method=self.pc_color_method, param=self.cfg.pc_color_param)
        self.cloud_pub.publish(points, timestamp)

        boxes3d = self.dv.transform_annos_to_boxes(info['annos'], with_eval_info=self.with_eval_info)
        if boxes3d.shape[0] != 0:
            boxes3d = self.dv.filter_boxes(idx, boxes3d, method=self.box_filter_method, param=self.cfg.box_filter_param)
            boxes3d_color = self.dv.render_boxes_color(idx, boxes3d, method=self.box_color_method, param=self.cfg.box_color_param)
        if boxes3d.shape[0] != 0:
            if self.with_eval_info:
                boxes3d, boxes3d_color = self.dv.merge_preds(boxes3d, boxes3d_color, info['preds'])

            corners_array = np.array([get_oriented_bounding_box_corners(b['box3d'][:3], b['box3d'][3:6], np.array([0, 0, b['box3d'][6]])) for b in boxes3d])
            lines = get_oriented_bounding_box_lines()
            self.boxes3d_pub.publish_boxes3d(corners_array, lines, timestamp, timer_period_ns=self.timer.timer_period_ns, colors=boxes3d_color)

        if self.pub_box_attr:
            boxes_text = [str(b['info']).replace(',', '\n') for b in boxes3d]
            self.box_attr_pub.publish_text(corners_array[:, 3], boxes_text, timestamp, timer_period_ns=self.timer.timer_period_ns, scale=[0.4, 0.4, 0.4])

        for key in self.cfg.cams_keys:
            img = info['image'][key]
            image = cv2.imread(f"{self.cfg.data_root}/{img['filepath']}")
            self.cams_image_pub[key].publish(image, timestamp)

            if self.pub_image_with_box:
                image_with_boxes = self.dv.get_image_with_box(copy.deepcopy(image), corners_array, boxes3d_color, img['l2p_mat']) if boxes3d.shape[0] != 0 else copy.deepcopy(image)
                self.cams_with_box_pub[key].publish(image_with_boxes, timestamp)

            if self.pub_image_with_points:
                image_with_points = self.dv.get_image_with_points(copy.deepcopy(image), points, img['l2p_mat'])
                self.cams_with_points_pub[key].publish(image_with_points, timestamp)

        if self.ctrl.play:
            self.ctrl.idx = (self.ctrl.idx + 1) % self.ctrl.len
        

def main(args=None):
    rclpy.init(args=args)

    node = OpenPCDetDatasetVisNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()