#!/usr/bin/env python3
#!coding=utf-8

import sys
import os
sys.path.append(os.path.split(sys.path[0])[0]+"/script/yolov5")
import rospy
from lidar import Radar
from camera import Camera,read_yaml
from cv_bridge import CvBridge
import cv2

import numpy as np
from yolov5 import detect

P = [
    [1882.9,  0,      1534.6],
    [0,       1882.9, 1076.9],
    [0,       0,      1]
    ]
K = [-0.1100,0.0890,0,0]
rospy.init_node('listener', anonymous=True)
cap = Camera()
bridge = CvBridge()
Camera.start()
api=detect.detectapi(weights='yolov5/weights/best.pt')
flag, frame = cap.read()
while (flag != True):
    flag, frame = cap.read()
    print(flag)

while True:
    # print("loop")
    flag,frame = cap.read()
    re_frame = bridge.imgmsg_to_cv2(frame, "bgr8")
    cam_distort = cv2.undistort(re_frame,np.array(P),np.array(K))
    result,names =api.detect([cam_distort])
    for cls,(x1,y1,x2,y2),conf in result[0][1]: #第一张图片的处理结果标签。
        #print(cls,x1,y1,x2,y2,conf)
        if cls==0 and conf >=0:
            cv2.rectangle(cam_distort,(x1,y1),(x2,y2),(0,255,0))
            cv2.putText(cam_distort,names[cls],(x1,y1-20),cv2.FONT_HERSHEY_DUPLEX,1.5,(255,0,0))
            print("%d,%d,%d,%d"%(x1,y1,x2,y2))
    resized = cv2.resize(cam_distort,(960,640))
    cv2.imshow("display",resized)
    cv2.waitKey(1)
# except:
#         print("session end.")
#         Camera.stop()
