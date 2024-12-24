import cv2
#!/usr/bin/env python3
#!coding=utf-8

import sys
import os
import math
# 系统必须库
import rospy
from lidar import Radar
from camera import Camera,read_yaml,SubCamera
# from location import locate_record
from cv_bridge import CvBridge
import cv2
import traceback
import argparse
from minimap import is_inside,car_classify,draw_area
# 用于相机去畸变
import numpy as np
# torch 识别框架
# from yolov5 import detect
# from yolov8 import infer_det
# 用于串口通信
from comtrans import ComTrans, procotol
from numba import jit, prange
from config import CLASSES, COLORS

from yolov8.models import TRTModule  # isort:skip

from pathlib import Path

import cv2
import numpy as np
import collections
import torch
from track import tracker
from yolov8.models.torch_utils import det_postprocess
from yolov8.models.utils import blob, letterbox, path_to_list
# 我方为红方
from datetime import datetime
def _locate_record(enemy,save = False,rvec = None,tvec = None):
    '''
    直接读取已储存的位姿，基于雷达每次位置变化不大
    这个函数也用来存储位姿

    :param camera_type:相机编号
    :param enemy:敌方编号
    :param save:读取还是存储
    :param rvec:当存储时，将旋转向量填入
    :param tvec:当存储时，将平移向量填入

    :return: （当为读取模型时有用）读取成功标志，旋转向量，平移向量
    '''
    LOCATION_SAVE_DIR = r"/home/qianzezhong/Documents/VSCode_Projects/lidar_new/src/hikrobot_camera/script/locate_info/"
    print("LOCATION_SAVE_DIR:",LOCATION_SAVE_DIR)
    max_order = -1
    max_file = None
    flag = False
    # 计算已存储的位姿文件中最大序号
    if not os.path.exists(LOCATION_SAVE_DIR):
        os.mkdir(LOCATION_SAVE_DIR)
    for f in os.listdir(LOCATION_SAVE_DIR):
        order,f_enemy,_ = f.split('_')
        order = int(order)
        f_enemy = int(f_enemy)
        # print("order:%d, f_enemy:%d"%(order,f_enemy))
        # 查询指定相机和敌方编号
        if f_enemy == enemy:
            if order > max_order:
                max_order = order
                max_file = f
    if save:
        print("****print为真****")
        filename = "{0}_{1}_{2}.txt".format(max_order+1,enemy,
                                                datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        with open(os.path.join(LOCATION_SAVE_DIR,filename),'w') as _log_f:
            _log_f.write("#rvec\n")
            _log_f.write(f"{float(rvec[0]):0.5f} {float(rvec[1]):0.5f} {float(rvec[2]):0.5f}\n")
            _log_f.write("#tvec\n")
            _log_f.write(f"{float(tvec[0]):0.5f} {float(tvec[1]):0.5f} {float(tvec[2]):0.5f}\n")
        # process_map()
        # build_lookup_table(0,enemy)
        # build_lookup_table(1,enemy)
    elif max_order > -1:
        # 读取模型，若文件不为空
        print("####读取模型阶段####")
        flag = True
        print(os.path.join(LOCATION_SAVE_DIR,max_file))
        pose = np.loadtxt(os.path.join(LOCATION_SAVE_DIR,max_file),delimiter=' ').reshape(2,3)
        rvec = pose[0]
        tvec = pose[1]

    return flag,rvec,tvec
Mode=0
def read_height_map(file_path):
    height_map = {}
    with open(file_path, 'r') as f:
        for line in f:
            coord_str, height_str = line.strip().split(': ')
            x, y = eval(coord_str)
            height = float(height_str)
            height_map[(x, y)] = height
    return height_map

def convert_to_numpy_array(height_map, x_range, y_range, step):
    x_samples = np.arange(x_range[0], x_range[1], step)
    y_samples = np.arange(y_range[0], y_range[1], step)
    height_array = np.zeros((len(y_samples), len(x_samples)))

    for i, y in enumerate(y_samples):
        for j, x in enumerate(x_samples):
            height_array[i, j] = height_map.get((float(str(f"{x:.2f}")), float(str(f"{y:.2f}"))), 0)  # 默认高度为0

    return height_array

def cluster_points_by_height(height_array, x_range, y_range, step):
    clusters = collections.defaultdict(list)
    for i, y in enumerate(np.arange(y_range[0], y_range[1], step)):
        for j, x in enumerate(np.arange(x_range[0], x_range[1], step)):
            height = height_array[i, j]
            clusters[height].append((x, y, height))
    sorted_clusters = sorted(clusters.items(), key=lambda item: item[0], reverse=True)
    return sorted_clusters

def project_points(points_3D, rvec, tvec, K_):
    points_3D = np.array(points_3D, dtype=np.float32)
    points_2D, _ = cv2.projectPoints(points_3D, rvec, tvec, K_, distCoeffs=None)
    return points_2D.reshape(-1, 2), points_3D

def create_lookup_table(image, clusters, rvec, tvec, K_, max_distance=30):
    image_shape = image.shape
    lookup_table = np.full((image_shape[0]//10+1, image_shape[1]//10+1, 3), 999, dtype=np.float32)
    print(rvec,tvec,K_)
    for height, points in clusters:
        print("cluster", height)
        points_2D, points_3D = project_points(points, rvec, tvec, K_)
        
        grid_y, grid_x = np.mgrid[0:image_shape[0]:10, 0:image_shape[1]:10]
        grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

        for point in grid_points:
            x, y = point
            if lookup_table[y//10, x//10, 0] == 999:
                distances = np.sqrt((points_2D[:, 0] - x)**2 + (points_2D[:, 1] - y)**2)
                min_distance_idx = np.argmin(distances)
                min_distance = distances[min_distance_idx]
                
                if min_distance < max_distance:
                    best_point = points_3D[min_distance_idx]
                    lookup_table[y//10, x//10] = best_point

    return lookup_table
# @jit(nopython=True, parallel=True)
# def calculate_distances(points_2D, grid_points):
#     distances = np.empty((grid_points.shape[0], points_2D.shape[0]), dtype=np.float32)
#     for i in prange(grid_points.shape[0]):
#         for j in prange(points_2D.shape[0]):
#             distances[i, j] = np.sqrt((points_2D[j, 0] - grid_points[i, 0])**2 + (points_2D[j, 1] - grid_points[i, 1])**2)
#     return distances

# def project_points(points_3D, rvec, tvec, K_):
#     points_3D = np.array(points_3D, dtype=np.float32)
#     points_2D, _ = cv2.projectPoints(points_3D, rvec, tvec, K_, distCoeffs=None)
#     return points_2D.reshape(-1, 2), points_3D

# def create_lookup_table(image, clusters, rvec, tvec, K_, max_distance=30):
#     image_shape = image.shape
#     lookup_table = np.full((image_shape[0]//5+1, image_shape[1]//5+1, 3), 999, dtype=np.float32)
    
#     for height, points in clusters:
#         print("cluster", height)
#         points_2D, points_3D = project_points(points, rvec, tvec, K_)
        
#         grid_y, grid_x = np.mgrid[0:image_shape[0]:10, 0:image_shape[1]:10]
#         grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
        
#         distances = calculate_distances(points_2D, grid_points)
        
#         for i, point in enumerate(grid_points):
#             x, y = point
#             if lookup_table[y//5, x//5, 0] == 999:
#                 min_distance_idx = np.argmin(distances[i])
#                 min_distance = distances[i, min_distance_idx]
                
#                 if min_distance < max_distance:
#                     best_point = points_3D[min_distance_idx]
#                     lookup_table[y//5, x//5] = best_point

#     return lookup_table

def lookup_real_world_coordinates(lookup_table, pixel_coord):
    x, y = pixel_coord
    coord = lookup_table[y//10, x//10]
    if coord[0] == 999:
        return (999, 999)
    else:
        return (coord[0], coord[1])

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        real_world_coord = lookup_real_world_coordinates(param, (x*2, y*2))
        print(f"Pixel coordinate: ({x*2}, {y*2}), Real world coordinate: {real_world_coord}")
def decompose_transform_matrix(T):
   # 提取旋转矩阵和平移向量
   R = T[:3, :3]
   tvec = T[:3, 3]

   # 将旋转矩阵转换为旋转向量
   rvec, _ = cv2.Rodrigues(R)

   return rvec, tvec
def build_lookup_table(_mode,IS_RED):
    Mode=_mode
    x_range = (0, 28.01)
    y_range = (-15, 0.01)
    step = 0.1

    height_map = read_height_map('/home/qianzezhong/Documents/VSCode_Projects/lidar_new/height_map.txt')
    height_array = convert_to_numpy_array(height_map, x_range, y_range, step)
    # IS_RED = True
    # ARMOR_CLASS=['B1','B2','B3','B4','B5','B7',
    # 'R1','R2','R3','R4','R5','R7']
    # 相机内参
    _,K_0,C_0,E_0,imgsz = read_yaml(0)
    _,K_1,C_1,E_1,imgsz1 = read_yaml(1)
    P = K_0
    K = C_0[0:4]
    P1 = K_1
    K1 = C_1[0:4]
    _,rvec,tvec = _locate_record(IS_RED,save=False)
    print(tvec)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.reshape(-1)
    if Mode==1:
        T = np.linalg.inv(T)
        T=T@E_0@np.linalg.inv(E_1)
        T = np.linalg.inv(T)
        rvec,tvec=decompose_transform_matrix(T)
    print("clustering...")
    clusters = cluster_points_by_height(height_array, x_range, y_range, step)
    if Mode==0:
        image = cv2.imread("/home/qianzezhong/Pictures/scene.png")
    else:
        image = cv2.imread("/home/qianzezhong/Pictures/scene_sub.png")
    resized_image = cv2.resize(image, (3072, 2048))
    print("setting up lookup table...")
    if Mode==0:
        lookup_table = create_lookup_table(resized_image, clusters, rvec, tvec, K_0)
    else:
        lookup_table = create_lookup_table(resized_image, clusters, rvec, tvec, K_1)
    print("done")
    # 存储查找表
    if Mode==0:
        np.save('/home/qianzezhong/Documents/VSCode_Projects/lidar_new/lookup_table_main.npy', lookup_table)
    else:
        np.save('/home/qianzezhong/Documents/VSCode_Projects/lidar_new/lookup_table_sub.npy', lookup_table)
if __name__ == "__main__":
    # 主程序
    x_range = (0, 28.01)
    y_range = (-15, 0.01)
    step = 0.1

    height_map = read_height_map('height_map.txt')
    height_array = convert_to_numpy_array(height_map, x_range, y_range, step)
    IS_RED=True
    _,K_0,C_0,E_0,imgsz = read_yaml(0)
    _,K_1,C_1,E_1,imgsz1 = read_yaml(1)
    P = K_0
    K = C_0[0:4]
    P1 = K_1
    K1 = C_1[0:4]
    _,rvec,tvec = _locate_record(int(not IS_RED),save=False)
    print(tvec)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.reshape(-1)
    if Mode==1:
        T = np.linalg.inv(T)
        T=T@E_0@np.linalg.inv(E_1)
        T = np.linalg.inv(T)
        rvec,tvec=decompose_transform_matrix(T)
    print("clustering...")
    clusters = cluster_points_by_height(height_array, x_range, y_range, step)
    if Mode==0:
        image = cv2.imread("/home/qianzezhong/Pictures/scene.png")
    else:
        image = cv2.imread("/home/qianzezhong/Pictures/scene_sub.png")
    resized_image = cv2.resize(image, (3072, 2048))
    print("setting up lookup table...")
    if Mode==0:
        lookup_table = create_lookup_table(resized_image, clusters, rvec, tvec, K_0)
    else:
        lookup_table = create_lookup_table(resized_image, clusters, rvec, tvec, K_1)
    print("done")
    # 存储查找表
    np.save('lookup_table.npy', lookup_table)

    # 设置鼠标回调
    cv2.namedWindow('Projected Image')
    cv2.setMouseCallback('Projected Image', mouse_callback, lookup_table)

    # 显示图像
    cv2.imshow('Projected Image', cv2.resize(resized_image, (resized_image.shape[1] // 2, resized_image.shape[0] // 2)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
