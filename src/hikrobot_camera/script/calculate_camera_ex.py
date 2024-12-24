import cv2
import numpy as np
from camera import read_yaml,Camera,SubCamera
import rospy
from cv_bridge import CvBridge
# 相机内参和畸变参数
camera_matrix_A = np.array([[2394.7    ,     0        ,      1695.17],     
    [0           ,   2352.09   ,     1063.33]    ,   
    [0         ,     0    ,          1]])
dist_coeffs_A = np.array([-0.1100, 0.0890, 0, 0, 0])

camera_matrix_B = np.array([[4991.32   ,     0,              1390.53],        
[0,              4890.63,        971.006],        
[0,              0,              1]])
dist_coeffs_B = np.array([-0.346427897727595, 1.114036359730677 ,0 ,0, 0])


object_points = np.array([
    [0, 0, 0],
    [0.03, 0, 0],
    [0.06, 0, 0],
    [0.15, 0, 0],
    [0.15, 0.03, 0],
    [0.15, 0.06, 0],
    [0.15, 0.21, 0],
    [0, 0.21, 0]
], dtype=np.float32)

# 鼠标回调函数，用于记录点击的点
points_A = []
points_B = []

def click_event_A(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points_A.append((x, y))
        cv2.circle(image_A, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image A', image_A)

def click_event_B(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points_B.append((x, y))
        cv2.circle(image_B, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image B', image_B)
import time
time.sleep(8)
rospy.init_node('four_point_listener', anonymous=True)
camera=Camera()
sub_camera=SubCamera()
bridge=CvBridge()
Camera.start()
SubCamera.start()
flag, frame = camera.read()
while (flag != True):
    flag, frame = camera.read()
image_A = bridge.imgmsg_to_cv2(frame, "bgr8")
flag, frame = sub_camera.read()
while (flag != True):
    flag, frame = sub_camera.read()
image_B = bridge.imgmsg_to_cv2(frame, "bgr8")
print(image_A.shape,image_B.shape)
# 读取图像
# image_A = cv2.imread('/home/qianzezhong/Pictures/main.bmp')
# image_B = cv2.imread('/home/qianzezhong/Pictures/sub.bmp')
cv2.namedWindow("Image A", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image A", 1280,780)
cv2.setWindowProperty("Image A",cv2.WND_PROP_TOPMOST,1)
cv2.moveWindow("Image A", 500,300)
cv2.imshow('Image A', image_A)
cv2.setMouseCallback('Image A', click_event_A)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.namedWindow("Image B", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image B", 1280,780)
cv2.setWindowProperty("Image B",cv2.WND_PROP_TOPMOST,1)
cv2.moveWindow("Image B", 500,300)
cv2.imshow('Image B', image_B)
cv2.setMouseCallback('Image B', click_event_B)
cv2.waitKey(0)

cv2.destroyAllWindows()

# 确保点数正确
if len(points_A) != len(object_points) or len(points_B) != len(object_points):
    print(f"You must click exactly {len(object_points)} points on each image.")
    exit()

# 转换为numpy数组
image_points_A = np.array(points_A, dtype=np.float32)
image_points_B = np.array(points_B, dtype=np.float32)

# 计算外参
_, rvec_A, tvec_A = cv2.solvePnP(object_points, image_points_A, camera_matrix_A, dist_coeffs_A)
_, rvec_B, tvec_B = cv2.solvePnP(object_points, image_points_B, camera_matrix_B, dist_coeffs_B)

# 旋转向量转换为旋转矩阵
R_A, _ = cv2.Rodrigues(rvec_A)
R_B, _ = cv2.Rodrigues(rvec_B)

# 构建齐次变换矩阵
extrinsics_A = np.hstack((R_A, tvec_A))
extrinsics_A = np.vstack((extrinsics_A, [0, 0, 0, 1]))

extrinsics_B = np.hstack((R_B, tvec_B))
extrinsics_B = np.vstack((extrinsics_B, [0, 0, 0, 1]))

print("Extrinsic matrix for Camera A:\n", extrinsics_A)
print("Extrinsic matrix for Camera B:\n", extrinsics_B)
