import numpy as np
from pu4c.det3d.utils import color_det_class25 as colormap
from pu4c.det3d.utils import get_oriented_bounding_box_lines, project_points_to_pixels
import cv2

colormap_bgr255 = (np.array(colormap)*255).astype(np.int64)[::-1]
class DatasetVisualizer:
    def __init__(self, infos, classes, default_cam=None) -> None:
        self.infos = infos
        self.classes = classes
        self.default_cam = default_cam
        self.map_cls_name2id = {cls:i for i, cls in enumerate(classes)}

    def render_point_cloud_color(self, idx, points, method='intensity', param=None):
        if method == 'intensity':
            points[:, 3] *= 255
        elif method == 'fov':
            param['l2p_mat'] = self.infos[idx]['image'][self.default_cam]['l2p_mat']
            _, _, mask = project_points_to_pixels(points, param['image_shape'], param['l2p_mat'])
            points[:, 3] *= 255
            points[mask, 3] = 0

    def render_boxes_color(self, idx, boxes3d, method='label', param=None):
        if method == 'label':
            colors = [colormap[b['label']] for b in boxes3d]
        elif method == 'difficulty':
            colors = [colormap[b['info']['difficulty']] for b in boxes3d]
        elif method == 'num_points_in_gt':
            num_points_in_gt = np.array([b['info']['num_points_in_gt'] for b in boxes3d])
            colors = [colormap[np.argmax(param['level'] >= num_pts)] for num_pts in num_points_in_gt]
        return np.array(colors)

    def filter_point_cloud(self, idx, points, method=[], param=None):
        if 'sensor' in method:
            points = points[points[:, 5] == param['sensor_id']]
        if 'fov' in method:
            param['l2p_mat'] = self.infos[idx]['image'][self.default_cam]['l2p_mat']
            _, _, mask = project_points_to_pixels(points[:, :3], param['image_shape'], param['l2p_mat'])
            points = points[mask]
        if 'range' in method:
            mask = (points[:, 0] >= param['limit_range'][0]) & (points[:, 0] <= param['limit_range'][3]) \
            & (points[:, 1] >= param['limit_range'][1]) & (points[:, 1] <= param['limit_range'][4])
            points = points[mask]

        return points

    def filter_boxes(self, idx, boxes3d, method=[], param=None):
        """
        过滤框，由于单纯的过滤框没有相应的过滤 info 会出错，故需要输出执行过滤的掩膜
        """
        if 'num_points' in method:
            num_points_in_gt = np.array([b['info']['num_points_in_gt'] for b in boxes3d])
            mask = num_points_in_gt >= param['min_points']
            boxes3d = boxes3d[mask]
        if 'fov' in method:
            param['l2p_mat'] = self.infos[idx]['image'][self.default_cam]['l2p_mat']
            centers = np.array([b['box3d'][:3] for b in boxes3d])
            _, _, mask = project_points_to_pixels(centers, param['image_shape'], param['l2p_mat'])
            boxes3d = boxes3d[mask]
        if 'range' in method:
            centers = np.array([b['box3d'][:3] for b in boxes3d])
            mask = ((centers[:, :2] >= param['limit_range'][:2]) & (centers[:, :2]  <= param['limit_range'][3:5])).all(axis=-1)
            boxes3d = boxes3d[mask]

        return boxes3d

    def get_image_with_points(self, image, points, l2p_mat, color=None):
        """
        Args:
            color: 默认 None 表示彩色渲染，否则传入纯色进行渲染，例如 (0, 0, 255) 
        """
        pixels, pixels_depth, mask = project_points_to_pixels(points, image.shape, l2p_mat)
        if color is None:
            mask = np.logical_and(mask, pixels_depth < 10*len(colormap_bgr255))
            color_levels = (pixels_depth[mask] / 10).astype(np.int32) # 每 10 米一级颜色，70 米共计 7 色
            for i, (x, y) in enumerate(pixels[mask]): 
                cv2.circle(image, center=(int(x), int(y)), radius=1, color=colormap_bgr255[color_levels[i]], thickness=-1)
        else:
            for x, y in pixels[mask]: # 图像坐标系，x-col 向右 y-row 向下，颜色 bgr，负数厚度表示绘制实心圆  
                cv2.circle(image, center=(int(x), int(y)), radius=1, color=(0, 0, 255), thickness=-1)

        return image

    def get_image_with_box(self, image, corners, boxes3d_color, l2p_mat):
        corners_pixels, _, mask = project_points_to_pixels(corners.reshape(-1, 3), image.shape, l2p_mat)
        corners_pixels, mask = corners_pixels.reshape(-1, 8, 2), mask.reshape(-1, 8)
        mask = [all(box_mask) for box_mask in mask]
        corners_pixels_array = corners_pixels[mask]

        lines = get_oriented_bounding_box_lines()
        rgbs = boxes3d_color[mask] * 255
        for i, pixels in enumerate(corners_pixels_array):
            bgr = (int(rgbs[i][2]), int(rgbs[i][1]), int(rgbs[i][0]))
            for line in lines:
                x0, y0 = int(pixels[line[0]][0]), int(pixels[line[0]][1])
                x1, y1 = int(pixels[line[1]][0]), int(pixels[line[1]][1])
                cv2.line(image, (x0, y0), (x1, y1), color=bgr, thickness=2)
        
        return image

    def transform_annos_to_boxes(self, annos, with_eval_info=False):
        """
        每个框及其附加信息汇总到一个字典里，便于后续处理
        """
        boxes3d = []
        for i, name in enumerate(annos['name']):
            if name in self.classes:
                label = self.map_cls_name2id[name] if not with_eval_info else -1
                box3d = annos['gt_boxes_lidar'][i]
                info = {
                    k:v[i] for k,v in annos.items()
                    if k not in ['gt_boxes_lidar']
                    }
                boxes3d.append({'label': label, 'box3d': box3d, 'info': info})

        return np.array(boxes3d)

    def merge_preds(self, boxes3d, boxes3d_color, preds=None):
        for model_id, pred_boxes3d in preds.items():
            if pred_boxes3d.shape[0] == 0:
                continue
            pred_boxes3d_color = self.render_boxes_color(0, pred_boxes3d, method='label', param=None)
            boxes3d = np.concatenate((boxes3d, pred_boxes3d))
            boxes3d_color = np.concatenate((boxes3d_color, pred_boxes3d_color))
            
        return boxes3d, boxes3d_color

    def decode_kitti_eval(self, gt_infos, eval_infos, model_id=0, uniform_color=False):
        """
        Args:
            uniform_color: 是否将该模型的预测用同一种颜色着色，用于同时查看多个模型预测时使用
        """
        lut = {gt_infos[i]['lidar']['frame_id']:i for i in range(len(gt_infos))}
        for eval_info in eval_infos:
            boxes3d = []
            for i, name in enumerate(eval_info['name']):
                label = self.map_cls_name2id[name] if not uniform_color else model_id
                box3d = eval_info['boxes_lidar'][i]
                boxes3d.append({'label': label, 'box3d': box3d, 'info': {'score': eval_info['score'][i]}})

            gt_idx = lut[eval_info['frame_id']]
            if 'preds' not in gt_infos[gt_idx]: gt_infos[gt_idx]['preds'] = {}
            gt_infos[gt_idx]['preds'][model_id] = np.array(boxes3d)

        return gt_infos


class Controller:
    def __init__(self, idx=0, len=1, play=False) -> None:
        self.idx = idx
        self.len = len
        self.play = play

    def onkey(self, msg, step=10):
        keycode = msg.data
        if keycode in ['w', 'W']:
            print(f"keycode w, idx - {step}")
            self.idx = (self.idx - step + self.len) % self.len
        elif keycode in ['s', 'S']:
            print(f"keycode s, idx + {step}")
            self.idx = (self.idx + step + self.len) % self.len
        elif keycode in ['a', 'A']:
            print(f"keycode a, idx - 1")
            self.idx = (self.idx - 1 + self.len) % self.len
        elif keycode in ['d', 'D']:
            print(f"keycode d, idx + 1")
            self.idx = (self.idx + 1 + self.len) % self.len
        elif keycode == ' ':
            print(f"keycode space, toogle play status")
            self.play = False if self.play else True

