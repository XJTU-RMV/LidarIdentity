import random

PC_STORE_DIR = "point_record" # 录制点云保存位置
LIDAR_TOPIC_NAME = "/livox/lidar" # 雷达PointCloud节点名称

CLASSES = ('car')

# colors for per classes
COLORS = {
    cls: [random.randint(0, 255) for _ in range(3)]
    for i, cls in enumerate(CLASSES)
}