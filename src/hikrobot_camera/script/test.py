#!/usr/bin/env python3
#!coding=utf-8

import sys
import os
import math
# 系统必须库
import rospy
from lidar import Radar
from camera import Camera,read_yaml,SubCamera
from location import locate_record
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
from config import CLASSES, COLORS

from yolov8.models import TRTModule  # isort:skip

from pathlib import Path

import cv2
import torch
from track import tracker
from yolov8.models.torch_utils import det_postprocess
from yolov8.models.utils import blob, letterbox, path_to_list
# 我方为红方

IS_RED = True
ARMOR_CLASS=['B1','B2','B3','B4','B5','B7',
'R1','R2','R3','R4','R5','R7']
# 相机内参
_,K_0,C_0,E_0,imgsz = read_yaml(0)
_,K_1,C_1,E_1,imgsz1 = read_yaml(1)
P = K_0
K = C_0[0:4]
P1 = K_1
K1 = C_1[0:4]
"""
good:
#rvec
1.27431 -1.45215 1.39121
#tvec
-5.88927 3.12631 -0.50191

"""
def lookup_real_world_coordinates(lookup_table, pixel_coord):
    x, y = pixel_coord
    coord = lookup_table[y//10, x//10]
    if coord[0] == 999:
        return (999, 999)
    else:
        return (coord[0], coord[1])
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
            # print(x,y)
            height_array[i, j] = height_map.get((float(str(f"{x:.2f}")), float(str(f"{y:.2f}"))), 0)  # 默认高度为0

    return height_array
def decompose_transform_matrix(T):
   # 提取旋转矩阵和平移向量
   R = T[:3, :3]
   tvec = T[:3, 3]

   # 将旋转矩阵转换为旋转向量
   rvec, _ = cv2.Rodrigues(R)

   return rvec, tvec
if __name__ == "__main__":
      
   # 主程序
   x_range = (0, 28.01)
   y_range = (-15, 0.01)
   step = 0.1

   height_map = read_height_map('height_map.txt')
   print(height_map)
   height_array = convert_to_numpy_array(height_map, x_range, y_range, step)
   print(height_array.shape)
   IS_RED = True
   _,rvec,tvec = locate_record(int(not IS_RED),save=False)
   print(tvec)
   T = np.eye(4)
   T[:3, :3] = cv2.Rodrigues(rvec)[0]
   T[:3, 3] = tvec.reshape(-1)
   T = np.linalg.inv(T)
   T=T@E_0@np.linalg.inv(E_1)
   T = np.linalg.inv(T)
   rvec,tvec=decompose_transform_matrix(T)

   image = cv2.imread("/home/qianzezhong/Pictures/scene_sub.png")
   resized_image = cv2.resize(image, (3072, 2048))
   point_3D = np.array([[x,y,height_array[int((y+15)*10),int(x*10)]] for x in np.arange(0,28.01,0.1) for y in np.arange(-15,0.01,0.1)]).astype(np.float32)  # 请填入具体的三维点坐标

   # 使用 cv2.projectPoints 将三维点投影到图像上
   point_2D, _ = cv2.projectPoints(point_3D, rvec, tvec, K_1, distCoeffs=None)

   print(point_2D.shape)
   # print(point_2D[1][0][1])
   # 绘制投影点在图像上
   for i in range(point_2D.shape[0]):
      point_2D2 = point_2D[i][0]  # 提取二维投影点坐标
      point_2D2 = (int(point_2D2[0]), int(point_2D2[1]))
      cv2.circle(resized_image, point_2D2, 5, (0, 255-point_3D[i][2]*255, point_3D[i][2]*255), -1)
      # cv2.putText(resized_image,f"{point_3D[i][0]} {point_3D[i][1]} {point_3D[i][2]:.2f}",point_2D2,0,0.5,(255,255,255),3)

   # 显示结果
   cv2.imshow('Projected Image', cv2.resize(resized_image,(resized_image.shape[1]//2,resized_image.shape[0]//2)))
   cv2.waitKey(0)
   cv2.destroyAllWindows()


