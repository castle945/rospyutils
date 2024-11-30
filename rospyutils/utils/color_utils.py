# 绿色: 真值
# 红橙色系(车): 吊车红色、拖车洋红、卡车暗红、公交车粉红、汽车橙色
# 黄色系(小型交通参与者): 行人黄色、摩托车葱黄、自行车棕黄
# 蓝色系(静态障碍): 围挡障碍蓝色、交通锥靛蓝
# 靛紫色系(其他): 动物等
det3d_colormap_class13 = [
    [255, 128, 0  ],  # 0  橙色 汽车
    [255, 180, 170],  # 1  粉红 车型级别二
    [160, 0  , 0  ],  # 2  暗红 车型级别三
    [255, 0  , 150],  # 3  洋红
    [255, 0  , 0  ],  # 4  红色
    [255, 255, 0  ],  # 5  黄色 行人
    [174, 112, 0  ],  # 6  棕黄 自行车
    [163, 217, 0  ],  # 7  葱黄 摩托车
    [0  , 0  , 255],  # 8  蓝色 围挡障碍
    [6  , 82 , 121],  # 9  靛蓝 交通锥
    [0  , 255, 255],  # 10 靛色 其他
    [255, 100, 255],  # 11 浅紫 其他
    [0  , 255, 0  ],  # 12 绿色 真值
]
colormap_ring7_rgb255 = [
    [0, 0, 255],   # 红
    [0, 128, 255], # 橙
    [0, 255, 255], # 黄
    [0, 255, 0],   # 绿
    [255, 0, 0],   # 蓝
    [130, 0, 75],  # 靛
    [211, 0, 148], # 紫
]

def get_det3d_colormap(categories: dict):
    colormap = [[] for _ in range(len(categories))]
    # 预定义所有已知类别的颜色索引
    colormap_indices = {
        'car': 0, 'Car': 0, 
        'bus': 1, 'Van': 1, 
        'truck': 2, 'Truck': 2, 
        'trailer': 3,
        'construction_vehicle': 4, 'Tram': 4,
        'pedestrian': 5, 'Pedestrian': 5, 
        'bicycle': 6, 'Cyclist': 6, 
        'motorcycle': 7,  
        'barrier': 8, 
        'traffic_cone': 9,
        'Misc': 10, 
        'Person_sitting': 11,
    }
    for name, id in categories.items():
        colormap[id] = det3d_colormap_class13[colormap_indices[name]]
    colormap.append(det3d_colormap_class13[-1])
    return colormap
        
