import numpy as np
from pu4c.cv.utils import get_oriented_bounding_box_lines, project_points_to_pixels
import cv2
from .utils import get_det3d_colormap, colormap_ring7_rgb255

class DatasetVisualizer:
    def __init__(self, infos, categories: dict, default_cam: str = None) -> None:
        self.infos = infos
        self.colormap_det3d = np.array(get_det3d_colormap(categories)) / 255.0  # 用于将类别标签的映射归一化 RGB 颜色
        self.colormap_ring7 = np.array(colormap_ring7_rgb255) / 255.0           # 用于简单的颜色映射如框内点数级别、检测框难度级别等
        self.colormap_ring7_bgr255 = np.array(colormap_ring7_rgb255).astype(np.int64)[::-1].tolist() # 用于 cv2 对像素着色
        self.default_cam = default_cam

    def render_point_cloud_color(self, idx, points, method='intensity', param=None):
        if method == 'intensity':
            points[:, 3] *= 255
        elif method == 'fov':
            param['lidar2img'] = self.infos[idx]['camera'][self.default_cam]['lidar2img']
            _, _, mask = project_points_to_pixels(points, param['image_shape'], param['lidar2img'])
            points[:, 3] *= 255 # 所有点按反射率着色
            points[mask, 3] = 0 # 投影点着红色，rviz 反射率 0 为红色

    def render_bbox_color(self, idx, bboxes_3d, method='label', param=None):
        if method == 'label':
            colors = [self.colormap_det3d[b['bbox_label']] for b in bboxes_3d]
        elif method == 'difficulty':
            colors = [self.colormap_ring7[b['bbox_attr']['difficulty']] for b in bboxes_3d]
        elif method == 'num_points':
            num_points_in_gt = np.array([b['bbox_attr']['num_points'] for b in bboxes_3d])
            colors = [self.colormap_ring7[np.argmax(param['min_points_list'] >= num_pts)] for num_pts in num_points_in_gt]
        else:
            colors = [self.colormap_det3d[-1] for _ in range(len(bboxes_3d))] # 默认为绿色真值
        return np.array(colors)

    def filter_point_cloud(self, idx, points, method=[], param=None):
        if 'sensor' in method:
            points = points[points[:, 5] == param['sensor_id']]
        if 'fov' in method:
            param['lidar2img'] = self.infos[idx]['camera'][self.default_cam]['lidar2img']
            _, _, mask = project_points_to_pixels(points[:, :3], param['image_shape'], param['lidar2img'])
            points = points[mask]
        if 'range' in method:
            mask = (points[:, 0] >= param['limit_range'][0]) & (points[:, 0] <= param['limit_range'][3]) \
            & (points[:, 1] >= param['limit_range'][1]) & (points[:, 1] <= param['limit_range'][4])
            points = points[mask]

        return points

    def filter_bbox(self, idx, bboxes_3d, method=[], param=None):
        if 'num_points' in method:
            num_points_in_gt = np.array([b['bbox_attr']['num_points'] for b in bboxes_3d])
            mask = num_points_in_gt >= param['min_points']
            bboxes_3d = bboxes_3d[mask]
        if 'fov' in method:
            param['lidar2img'] = self.infos[idx]['camera'][self.default_cam]['lidar2img']
            centers = np.array([b['bbox_3d'][:3] for b in bboxes_3d])
            _, _, mask = project_points_to_pixels(centers, param['image_shape'], param['lidar2img'])
            bboxes_3d = bboxes_3d[mask]
        if 'range' in method:
            centers = np.array([b['bbox_3d'][:3] for b in bboxes_3d])
            mask = ((centers[:, :2] >= param['limit_range'][:2]) & (centers[:, :2]  <= param['limit_range'][3:5])).all(axis=-1)
            bboxes_3d = bboxes_3d[mask]

        return bboxes_3d

    def get_image_with_points(self, image, points, lidar2img, color=None):
        """
        Args:
            color: 默认 None 表示彩色渲染，否则传入纯色进行渲染，例如 (0, 0, 255) 
        """
        pixels, pixels_depth, mask = project_points_to_pixels(points, image.shape, lidar2img)
        if color is None:
            mask = np.logical_and(mask, pixels_depth < 10*len(self.colormap_ring7_bgr255))
            color_levels = (pixels_depth[mask] / 10).astype(np.int32) # 每 10 米一级颜色，70 米共计 7 色
            for i, (x, y) in enumerate(pixels[mask]): 
                cv2.circle(image, center=(int(x), int(y)), radius=1, color=self.colormap_ring7_bgr255[color_levels[i]], thickness=-1)
        else:
            for x, y in pixels[mask]: # 图像坐标系，x-col 向右 y-row 向下，颜色 bgr，负数厚度表示绘制实心圆  
                cv2.circle(image, center=(int(x), int(y)), radius=1, color=(0, 0, 255), thickness=-1)

        return image

    def get_image_with_box(self, image, corners, bboxes_3d_color_rgb1, lidar2img):
        corners_pixels, _, mask = project_points_to_pixels(corners.reshape(-1, 3), image.shape, lidar2img)
        corners_pixels, mask = corners_pixels.reshape(-1, 8, 2), mask.reshape(-1, 8)
        mask = [all(box_mask) for box_mask in mask]
        corners_pixels_array = corners_pixels[mask]

        lines = get_oriented_bounding_box_lines()
        rgbs = bboxes_3d_color_rgb1[mask] * 255
        for i, pixels in enumerate(corners_pixels_array):
            bgr = (int(rgbs[i][2]), int(rgbs[i][1]), int(rgbs[i][0]))
            for line in lines:
                x0, y0 = int(pixels[line[0]][0]), int(pixels[line[0]][1])
                x1, y1 = int(pixels[line[1]][0]), int(pixels[line[1]][1])
                cv2.line(image, (x0, y0), (x1, y1), color=bgr, thickness=2)
        
        return image

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