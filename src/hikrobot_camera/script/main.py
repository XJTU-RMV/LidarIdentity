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
import time
class Timer:
    def __init__(self):
        self.now_time=time.time()
        # print("TimerStart.")
    def update(self,text):
        # print(text,time.time()-self.now_time)
        self.now_time=time.time()
# 我方为红方
IS_RED = True
IS_SHOW_MAP=False 
lookup_table_main=np.load('/home/qianzezhong/Documents/VSCode_Projects/lidar_new/lookup_table_main.npy')
lookup_table_sub=np.load('/home/qianzezhong/Documents/VSCode_Projects/lidar_new/lookup_table_sub.npy')
ARMOR_CLASS=['B1','B2','B3','B4','B5','B7',
'R1','R2','R3','R4','R5','R7']
# 相机内参
_,K_0,C_0,E_0,imgsz = read_yaml(0)
_,K_1,C_1,E_1,imgsz1 = read_yaml(1)
P = K_0
K = C_0[0:4]
P1 = K_1
K1 = C_1[0:4]
def lookup_real_world_coordinates(lookup_table, pixel_coord):
    x, y = pixel_coord
    if y//10< 0 or y//10 > lookup_table.shape[0]:
        return (np.nan, np.nan,np.nan)
    if x//10<0 or x//10 > lookup_table.shape[1]:
        return (np.nan, np.nan,np.nan)
    coord = lookup_table[y//10, x//10]
    if coord[0] == 999:
        return (np.nan, np.nan,np.nan)
    else:
        return (coord[0], coord[1],coord[2])
def intersection_over_min_area(box1, box2):
    """
    计算两个边界框的交集面积除以两个边界框面积的最小值
    """
    # 计算交集面积
    x1 = torch.max(box1[:,0], box2[:,0])
    y1 = torch.max(box1[:,1], box2[:,1])
    x2 = torch.min(box1[:,2], box2[:,2])
    y2 = torch.min(box1[:,3], box2[:,3])

    intersection_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # 计算边界框面积
    area1 = (box1[:,2] - box1[:,0]) * (box1[:,3] - box1[:,1])
    area2 = (box2[:,2] - box2[:,0]) * (box2[:,3] - box2[:,1])

    # 计算交集面积除以两个边界框面积的最小值
    min_area = torch.min(area1, area2)
    intersection_over_min_area = intersection_area / min_area

    return intersection_over_min_area.item()
def calculate_iou(box1, box2):
    """
    计算两个边界框之间的 IoU (Intersection over Union)
    """
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    union = area1 + area2 - intersection

    iou = intersection / union
    return iou

def filter_bboxes(bboxes, scores, labels):
    """
    过滤边界框并保留具有最高分数的边界框
    """
    filtered_bboxes = []
    filtered_scores = []
    filtered_labels = []
    
    for i in range(bboxes.size(0)):

        bbox = bboxes[i]
        score = scores[i]
        label = labels[i]
        keep_bbox = True
        # print(bbox)
        # if bbox[2]-bbox[0]>300 or bbox[3]-bbox[1]>300: # 超大框框
        #     score=0
        #     keep_bbox=False
        
        for j in range(bboxes.size(0)):
            if i==j:
                continue
            other_bbox = bboxes[j].unsqueeze(0)
            iou = calculate_iou(bbox.unsqueeze(0), other_bbox)
            iom = intersection_over_min_area(bbox.unsqueeze(0), other_bbox)
            # print("iou,iom",iou,iom)
            if iou > 0.65 or iom > 0.8:
                if scores[j] > score:
                    print("filted")
                    keep_bbox = False
                    break
        if keep_bbox:
            filtered_bboxes.append(bbox)
            filtered_scores.append(score)
            filtered_labels.append(label)
    return torch.stack(filtered_bboxes), torch.tensor(filtered_scores), torch.tensor(filtered_labels)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    parser.add_argument('--color',
                        type=int,
                        default=1)
    parser.add_argument('--record',
                        type=int,
                        default=0)
    parser.add_argument('--armor_engine', type=str, help='Armor Engine file')
    args = parser.parse_args()
    return args

# 转移矩阵
_,rvec,tvec = locate_record(int(not IS_RED),save=False)
T = np.eye(4)
T[:3, :3] = cv2.Rodrigues(rvec)[0]
T[:3, 3] = tvec.reshape(-1)
T = np.linalg.inv(T)
EXIT_FLAG=False
# 用于发送串口
# comtrans = ComTrans("/dev/ttyACM0")
com_list = [0 for i in range(10)]
def processCamera(frame,Engine,device,ArmorEngine,ImageSize:list[int,int],ArmorImageSize:list[int,int],sublidar:Radar):# 处理图像，ImageSize为[W,H]
    bgr=bridge.imgmsg_to_cv2(frame, "bgr8")
    bgr_distort = cv2.undistort(bgr,np.array(P),np.array(K))  # 去畸变
    
    bgr_distort_copy = bgr_distort.copy()
    
    
    bgr_distort, ratio, dwdh = letterbox(bgr_distort, (ImageSize[0], ImageSize[1]))
    # print(bgr_distort.shape)
    rgb = cv2.cvtColor(bgr_distort, cv2.COLOR_BGR2RGB)
    tensor = blob(rgb, return_seg=False)
    dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
    tensor = torch.asarray(tensor, device=device)
    # inference
    data = Engine(tensor)
    # print(data)
    bboxes, scores, labels = det_postprocess(data)
    if bboxes.numel() == 0:
        # if no bounding box
        # print(f'camera no object!')
        ...
    else:
        ...
        bboxes, scores, labels = filter_bboxes(bboxes,scores,labels)
    origin_bboxes=bboxes.clone()
    bboxes -= dwdh
    bboxes /= ratio
    cps=[]
    armors=[]
    for (bbox, score, label,ob) in zip(bboxes, scores, labels,origin_bboxes):
            # print(bbox)
            bbox = bbox.round().int().tolist()
            bbox[0]=max(bbox[0],0)
            bbox[1]=max(bbox[1],0)
            if bbox[2]-bbox[0] > 600 or bbox[3]-bbox[1]>600: # 大框框
                continue
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            
            
            if ob[0]==ob[2] or ob[1]==ob[3]:
                #armors.append([-1,0])
                continue
            armor_distort = (bgr_distort_copy)[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
            # armor_distort = (bgr_distort)[int(ob[1]):int(ob[3]),int(ob[0]):int(ob[2])]
            if armor_distort.shape[0]==0:
                continue
            try:
                armor_distort, armor_ratio, armor_dwdh = letterbox(armor_distort, (ArmorImageSize[0], ArmorImageSize[1]))
            except:
                continue
            # cv2.imshow("a",armor_distort)
            rgb = cv2.cvtColor(armor_distort, cv2.COLOR_BGR2RGB)
            armor_tensor = blob(rgb, return_seg=False)
            armor_dwdh = torch.asarray(armor_dwdh * 2, dtype=torch.float32, device=device)
            armor_tensor = torch.asarray(armor_tensor, device=device)
            armor_data = ArmorEngine(armor_tensor)
            
            armor_bboxes, armor_scores, armor_labels = det_postprocess(armor_data)
            armor_bboxes-=armor_dwdh
            armor_bboxes/=armor_ratio
            armor_bboxes+=torch.Tensor([bbox[0],bbox[1],bbox[0],bbox[1]]).cuda()
            # armor_bboxes+=dwdh
            #armor_bboxes/=ratio
            mx_score=0
            armor_box_pos=[]
            av_labels=[]
            
            for ii,(armor_bbox, armor_score, armor_label) in enumerate(zip(armor_bboxes, armor_scores, armor_labels)):
                try:
                    armor_bbox = armor_bbox.round().int().tolist()
                    if (armor_bbox[2]-armor_bbox[0])*(armor_bbox[3]-armor_bbox[1])<1:
                        continue
                    # mx_score+=armor_score.item()
                    # armor_box_pos.append(np.array(armor_bbox))
                    label_name=ARMOR_CLASS[int(armor_label.cpu().item())]
                
                    x1, y1, x2, y2 = armor_bbox
                    rectangle_region = bgr_distort_copy[y1:y2, x1:x2]
                    red_pixels = cv2.inRange(rectangle_region, np.array([0, 0, 100]), np.array([100, 100, 255]))
                    blue_pixels = cv2.inRange(rectangle_region,np.array([60, 0, 0]), np.array([255, 150, 150]))
                    red_pixel_count = cv2.countNonZero(red_pixels)
                    blue_pixel_count = cv2.countNonZero(blue_pixels)
                    print(red_pixel_count,blue_pixel_count,label_name)
                    if label_name[0]=='B' and blue_pixel_count<5:#防止颜色识别错误
                        armor_scores[ii]=0
                        continue
                    if label_name[0]=='R' and red_pixel_count<5:
                        armor_scores[ii]=0
                        continue
                    if not int(armor_label.cpu().item()) in av_labels:
                        av_labels.append(int(armor_label.cpu().item()))
                    cv2.rectangle(bgr_distort_copy, armor_bbox[:2], armor_bbox[2:], color, 2)
                    cv2.putText(bgr_distort_copy,f"{label_name}:{armor_score}",(armor_bbox[0], armor_bbox[1] - 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, [225, 255, 255],
                                thickness=2)
                except:
                    print("我不造啊")
            now_cps=[]
            if armor_scores.shape[0]>0:
                for av_l in av_labels:
                    mx_score=0
                    mx_bbox=None
                    for (armor_bbox, armor_score, armor_label) in zip(armor_bboxes, armor_scores, armor_labels):
                        armor_bbox = armor_bbox.round().int().tolist()
                        if (armor_bbox[2]-armor_bbox[0])*(armor_bbox[3]-armor_bbox[1])<1 or len(av_labels)>1 and (armor_bbox[2]-armor_bbox[0])/(armor_bbox[3]-armor_bbox[1])<0.6:# 太侧边的识别不准 咱不要.判断标准:长宽比小于0.8
                            continue
                        armor_label=int(armor_label.cpu().item())
                        armor_score=armor_score.cpu().item()
                        if armor_label==av_l and armor_score>mx_score:
                            mx_score=armor_score
                            mx_bbox=armor_bbox
                    if mx_bbox != None:
                        armor_box_pos=np.array(mx_bbox).astype(int)
                        # fixed_armor_x=int(((armor_box_pos[0]+armor_box_pos[2])/2+(bbox[0]+bbox[2])/2)/2-(armor_box_pos[2]-armor_box_pos[0])/2)
                        # fixed_armor_y=int(((armor_box_pos[1]+armor_box_pos[3])/2+(bbox[1]+bbox[3])/2)/2-(armor_box_pos[3]-armor_box_pos[1])/2)
                        # fixed使得装甲版更加靠近中心
                        # cp = sublidar.detect_depth((fixed_armor_x,fixed_armor_y,armor_box_pos[2]-armor_box_pos[0],
                        #         armor_box_pos[3]-armor_box_pos[1])).reshape(-1)
                        # cp = sublidar.detect_depth((armor_box_pos[0],armor_box_pos[1],armor_box_pos[2]-armor_box_pos[0],
                        #         armor_box_pos[3]-armor_box_pos[1])).reshape(-1)
                        # E_0@L=C0 T@C0=W
                        # E_1@L=C1 ?@C1=W
                        # cp = ( (T@E_0@np.linalg.inv(E_1)) @ np.concatenate(
                        #         [np.concatenate([cp[:2], np.ones(1)], axis=0) * cp[2], np.ones(1)], axis=0))[:3]
                        # print(armor_box_pos)
                        pt=((armor_box_pos[0]+armor_box_pos[2])//2,int((armor_box_pos[3]*0.8+armor_box_pos[1]*0.2)))
                        # print(pt)
                        # cv2.circle(bgr_distort_copy,pt,5,(0,255,0),-1)
                        cp=lookup_real_world_coordinates(lookup_table_sub,pt)
                        # print(cp)
                        if math.isnan(cp[0]):
                            print("nan")
                            continue
                        if mx_score<min(0.4+len(av_labels)*0.1,0.8):
                            armors.append([-1,0])
                        else:
                            armors.append([av_l,mx_score])
                        
                        cps.append(cp)
                        now_cps.append(cp)
            else:
                
                pt=((bbox[0]+bbox[2])//2,int((bbox[3]*0.8+bbox[1]*0.2)))
                # print(pt)
                # cv2.circle(bgr_distort_copy,pt,5,(0,255,0),-1)
                cp=lookup_real_world_coordinates(lookup_table_sub,pt)
                # cp = sublidar.detect_depth((bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1])).reshape(-1)
                # cp = ( (T@E_0@np.linalg.inv(E_1)) @ np.concatenate(
                #         [np.concatenate([cp[:2], np.ones(1)], axis=0) * cp[2], np.ones(1)], axis=0))[:3]
                if math.isnan(cp[0]):
                    continue
                armors.append([-1,0])
                cps.append(cp)
                now_cps.append(cp)
            for _idx,cp in enumerate(now_cps):
                if not math.isnan(cp[0]): 
                        cv2.putText(bgr_distort_copy,
                                    f'{cls}:{score:.3f}:{cp[0]:.3f}:{cp[1]:.3f}:{cp[2]:.3f}', (bbox[0], bbox[1] - 2-_idx*20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.75, [225, 255, 255],
                                    thickness=2)
                else:
                    cv2.putText(bgr_distort_copy,
                                f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2-_idx*20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, [225, 255, 255],
                                thickness=2)
            cv2.rectangle(bgr_distort_copy, bbox[:2], bbox[2:], color, 2)
    if IS_SHOW_MAP:
        for i in range(0,lookup_table_sub.shape[0],3):
            for j in range(0,lookup_table_sub.shape[1],3):
                if lookup_table_sub[i][j][0]!=999:
                    cv2.circle(bgr_distort_copy,(j*10,i*10),1,(0,lookup_table_sub[i][j][0]*8,255-lookup_table_sub[i][j][0]*8),1)
    return bgr_distort_copy,cps,armors
    
def main_camera_gen(cap:cv2.VideoCapture,bridge:CvBridge,lidar:Radar,sublidar:Radar,args:argparse,subcap:cv2.VideoCapture=None): # ':'类型注解
    # .engine模型初始化
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    ArmorEngine=TRTModule(args.armor_engine,device)
    H, W = Engine.inp_info[0].shape[-2:]
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
    ArmorEngine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
    armorH,armorW=ArmorEngine.inp_info[0].shape[-2:]
    flag, frame = cap.read()
    while (flag != True):
        flag, frame = cap.read()
    tr=tracker(IS_RED)
    
    #try:
    while True:
        timer=Timer()
        # com = com_list.copy()
        
        flag,frame = cap.read()
        timer.update("GetRosCamera")
        if subcap != None:
            subflag,subframe=subcap.read()
            processedImage,sub_cps,sub_armors=processCamera(subframe,Engine,device,ArmorEngine,(W,H),(armorW,armorH),sublidar)
            sub_resized=cv2.resize(processedImage,(1536,1024))
            cv2.imshow("subcamera",sub_resized)
        timer.update("SubCameraProcessing")
        print("########")
        bgr = bridge.imgmsg_to_cv2(frame, "bgr8")
        timer.update("RosToCV2")
        bgr_distort = cv2.undistort(bgr,np.array(P),np.array(K))  # 去畸变
        bgr_distort_copy = bgr_distort.copy()
        bgr_distort, ratio, dwdh = letterbox(bgr_distort, (W, H))
        rgb = cv2.cvtColor(bgr_distort, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)
        timer.update("NumpyToTensor")
        # inference
        # print(tensor.shape)
        
        data = Engine(tensor)
        timer.update("Model Running")

        bboxes, scores, labels = det_postprocess(data)
        origin_bboxes=bboxes.clone()
        if bboxes.numel() == 0:
            # if no bounding box
            # print(f'no object!')
            ...
        else:
            ...
            bboxes, scores, labels = filter_bboxes(bboxes,scores,labels)
        bboxes -= dwdh
        bboxes /= ratio
        # num_point = 0
        # print(bboxes,scores,labels)
        cps=[]
        armors=[]
        for (bbox, score, label,ob) in zip(bboxes, scores, labels,origin_bboxes):
            bbox = bbox.round().int().tolist()
            bbox[0]=max(bbox[0],0)
            bbox[1]=max(bbox[1],0)
            # print(bbox)
            if bbox[2]-bbox[0] > 600 or bbox[3]-bbox[1]>600: # 大框框
                continue
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            
            
            if ob[0]==ob[2] or ob[1]==ob[3]:
                #armors.append([-1,0])
                continue
            armor_distort = (bgr_distort_copy)[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
            # armor_distort = (bgr_distort)[int(ob[1]):int(ob[3]),int(ob[0]):int(ob[2])]
            if armor_distort.shape[0]==0:
                continue
            try:
                armor_distort, armor_ratio, armor_dwdh = letterbox(armor_distort, (armorW, armorH))
            except:
                continue
            # cv2.imshow("a",armor_distort)
            rgb = cv2.cvtColor(armor_distort, cv2.COLOR_BGR2RGB)
            armor_tensor = blob(rgb, return_seg=False)
            armor_dwdh = torch.asarray(armor_dwdh * 2, dtype=torch.float32, device=device)
            armor_tensor = torch.asarray(armor_tensor, device=device)
            armor_data = ArmorEngine(armor_tensor)
            
            armor_bboxes, armor_scores, armor_labels = det_postprocess(armor_data)
            armor_bboxes-=armor_dwdh
            armor_bboxes/=armor_ratio
            armor_bboxes+=torch.Tensor([bbox[0],bbox[1],bbox[0],bbox[1]]).cuda()
            # armor_bboxes+=dwdh
            #armor_bboxes/=ratio
            mx_score=0
            armor_box_pos=[]
            av_labels=[]
            
            for ii,(armor_bbox, armor_score, armor_label) in enumerate(zip(armor_bboxes, armor_scores, armor_labels)):
                try:
                    armor_bbox = armor_bbox.round().int().tolist()
                    if (armor_bbox[2]-armor_bbox[0])*(armor_bbox[3]-armor_bbox[1])<1:
                        continue

                    if (armor_bbox[2]-armor_bbox[0])<20: # 短焦不处理太小的装甲版
                        continue
                    label_name=ARMOR_CLASS[int(armor_label.cpu().item())]
                    # print(label_name)
                    # if (armor_bbox[2]-armor_bbox[0])<40 and (label_name=='B7' or label_name=='R7'):
                    #     continue
                    # mx_score+=armor_score.item()
                    # armor_box_pos.append(np.array(armor_bbox))
                    
                
                    x1, y1, x2, y2 = armor_bbox
                    rectangle_region = bgr_distort_copy[y1:y2, x1:x2]
                    # print(x1,x2,y1,y2)
                    red_pixels = cv2.inRange(rectangle_region, np.array([0, 0, 100]), np.array([100, 100, 255]))
                    blue_pixels = cv2.inRange(rectangle_region,np.array([60, 0, 0]), np.array([255, 150, 150]))
                    red_pixel_count = cv2.countNonZero(red_pixels)
                    blue_pixel_count = cv2.countNonZero(blue_pixels)
                    if label_name[0]=='B' and blue_pixel_count<5:#防止颜色识别错误
                        armor_scores[ii]=0
                        continue
                    if label_name[0]=='R' and red_pixel_count<5:
                        armor_scores[ii]=0
                        continue
                    if not int(armor_label.cpu().item()) in av_labels:
                        av_labels.append(int(armor_label.cpu().item()))
                    cv2.rectangle(bgr_distort_copy, armor_bbox[:2], armor_bbox[2:], color, 2)
                    
                    cv2.putText(bgr_distort_copy,f"{label_name}:{armor_score}",(armor_bbox[0], armor_bbox[1] - 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, [225, 255, 255],
                                thickness=2)
                except:
                    print("我不造啊")
            now_cps=[]
            if armor_scores.shape[0]>0:
                for av_l in av_labels:
                    mx_score=0
                    mx_bbox=None
                    for (armor_bbox, armor_score, armor_label) in zip(armor_bboxes, armor_scores, armor_labels):
                        armor_bbox = armor_bbox.round().int().tolist()
                        if (armor_bbox[2]-armor_bbox[0])*(armor_bbox[3]-armor_bbox[1])<1 or len(av_labels)>1 and (armor_bbox[2]-armor_bbox[0])/(armor_bbox[3]-armor_bbox[1])<0.6:# 太侧边的识别不准 咱不要.判断标准:长宽比小于0.8
                            print("filted",armor_label)
                            continue

                        armor_label=int(armor_label.cpu().item())
                        armor_score=armor_score.cpu().item()
                        
                        if armor_label==av_l and armor_score>mx_score:
                            mx_score=armor_score
                            mx_bbox=armor_bbox
                    if mx_bbox != None:
                        armor_box_pos=np.array(mx_bbox).astype(int)
                        # fixed_armor_x=int(((armor_box_pos[0]+armor_box_pos[2])/2+(bbox[0]+bbox[2])/2)/2-(armor_box_pos[2]-armor_box_pos[0])/2)
                        # fixed_armor_y=int(((armor_box_pos[1]+armor_box_pos[3])/2+(bbox[1]+bbox[3])/2)/2-(armor_box_pos[3]-armor_box_pos[1])/2)
                        # fixed使得装甲版更加靠近中心
                        pt=((armor_box_pos[0]+armor_box_pos[2])//2,int((armor_box_pos[3]*0.8+armor_box_pos[1]*0.2)))
                        # print(pt)
                        # cv2.circle(bgr_distort_copy,pt,5,(0,255,0),-1)
                        cp=lookup_real_world_coordinates(lookup_table_main,pt)
                        # print(armor_box_pos)
                        # cp = lidar.detect_depth((fixed_armor_x,fixed_armor_y,armor_box_pos[2]-armor_box_pos[0],
                        #         armor_box_pos[3]-armor_box_pos[1])).reshape(-1)
                        # cp = sublidar.detect_depth((armor_box_pos[0],armor_box_pos[1],armor_box_pos[2]-armor_box_pos[0],
                        #         armor_box_pos[3]-armor_box_pos[1])).reshape(-1)

                        # cp = ( (T) @ np.concatenate(
                        #         [np.concatenate([cp[:2], np.ones(1)], axis=0) * cp[2], np.ones(1)], axis=0))[:3]
                        

                        if math.isnan(cp[0]):
                            print("nan")
                            continue
                        if mx_score<min(0.4+len(av_labels)*0.1,0.8):
                            armors.append([-1,0])
                        else:
                            armors.append([av_l,mx_score])
                        
                        cps.append(cp)
                        now_cps.append(cp)
            else:
                pt=((bbox[0]+bbox[2])//2,int((bbox[3]*0.8+bbox[1]*0.2)))
                # print(pt)
                # cv2.circle(bgr_distort_copy,pt,5,(0,255,0),-1)
                cp=lookup_real_world_coordinates(lookup_table_main,pt)
                # cp = sublidar.detect_depth((bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1])).reshape(-1)
                # cp = ( (T@E_0@np.linalg.inv(E_1)) @ np.concatenate(
                #         [np.concatenate([cp[:2], np.ones(1)], axis=0) * cp[2], np.ones(1)], axis=0))[:3]
                if math.isnan(cp[0]):
                    continue
                armors.append([-1,0])
                cps.append(cp)
                now_cps.append(cp)
            for _idx,cp in enumerate(now_cps):
                if not math.isnan(cp[0]): 
                        cv2.putText(bgr_distort_copy,
                                    f'{cls}:{score:.3f}:{cp[0]:.3f}:{cp[1]:.3f}:{cp[2]:.3f}', (bbox[0], bbox[1] - 2-_idx*20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.75, [225, 255, 255],
                                    thickness=2)
                else:
                    cv2.putText(bgr_distort_copy,
                                f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2-_idx*20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, [225, 255, 255],
                                thickness=2)
            cv2.rectangle(bgr_distort_copy, bbox[:2], bbox[2:], color, 2)
            
            # cv2.putText(bgr_distort_copy,
            #             f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             0.75, [225, 255, 255],
            #             thickness=2)
        if subcap != None:#主次相机相同车去重
            temp_cps=[]
            temp_armors=[]
            for i in range(len(sub_armors)):
                _flag=True
                # for j in range(len(armors)):
                #     if sub_armors[i][0] != -1 and sub_armors[i][0]==armors[j][0]:
                #         _flag=False
                #         # cps[j][0]=(cps[j][0]*armors[j][1]+sub_cps[i][0]*sub_armors[i][1])/(armors[j][1]+sub_armors[i][1])
                #         # cps[j][1]=(cps[j][1]*armors[j][1]+sub_cps[i][1]*sub_armors[i][1])/(armors[j][1]+sub_armors[i][1])
                if _flag:
                    temp_cps.append(sub_cps[i])
                    temp_armors.append(sub_armors[i])
            for i in range(len(armors)):
                _flag=True
                for j in range(len(sub_armors)):
                    if armors[i][0] != -1 and sub_armors[j][0]==armors[i][0]:
                        # sub_cps[j][0]= (sub_cps[j][0]*sub_armors[j][1]+cps[i][0]*armors[i][1])/(sub_armors[j][1]+armors[i][1])
                        # sub_cps[j][1]=(sub_cps[j][1]*sub_armors[j][1]+cps[i][1]*armors[i][1])/(sub_armors[j][1]+armors[i][1])
                        _flag=False
                    
                if _flag:
                    temp_cps.append(cps[i])
                    temp_armors.append(armors[i])
            #print(len(temp_armors),len(temp_cps))
            tr.update(temp_cps,temp_armors)
        else:
            tr.update(cps,armors)
        timer.update("Other Processing(Serial,Tracking)")
        if IS_SHOW_MAP:
            for i in range(0,lookup_table_main.shape[0],3):
                for j in range(0,lookup_table_main.shape[1],3):
                    if lookup_table_main[i][j][0]!=999:
                        cv2.circle(bgr_distort_copy,(j*10,i*10),1,(0,lookup_table_main[i][j][0]*8,255-lookup_table_main[i][j][0]*8),1)

        resized = cv2.resize(bgr_distort_copy,(1536,1024))
        cv2.imshow('result', resized)
        timer.update("ShowPicture")
        key = cv2.waitKey(1)

        if key == 113: # 赛场最好关掉，防止误触
            EXIT_FLAG=True
            tr.stop()
            
            exit(0)
        timer.update("END")
        # cv2.waitKey(1000)
        # for cls,(x1,y1,x2,y2),conf in result[0][1]: #第一张图片的处理结果标签。
        #     if cls==0 and conf >=0.5:
        #         cam_copy = cam_distort.copy()
        #         chopped = cam_copy[y1:y2,x1:x2]
        #         if car_classify(chopped,red=not IS_RED):
        #             cv2.rectangle(cam_distort,(x1,y1),(x2,y2),(0,255,0),5)
        #             cv2.putText(cam_distort,names[cls],(x1,y1-20),cv2.FONT_HERSHEY_DUPLEX,1.5,(255,0,0))
        #             w = x2-x1
        #             h = y2-y1
        #             cp = lidar.detect_depth((x1,y1,w,h)).reshape(-1)
        #             cp = ( T @ np.concatenate(
        #             [np.concatenate([cp[:2], np.ones(1)], axis=0) * cp[2], np.ones(1)], axis=0))[:3]
        #             if not math.isnan(cp[0]):       
        #                 map_x = int(cp[0] / 28 * 1150)
        #                 if map_x > 1150:
        #                     map_x = 1150
        #                 if map_x < 0:
        #                     map_x = 0
        #                 map_y = -int(cp[1] / 15 * 616)
        #                 if map_y > 616:
        #                     map_y = 616
        #                 if map_y < 0:
        #                     map_y = 0
        #                 com[9 - num_point*2 - 1] = round(min(max(cp[0], 0), 28), 3)
        #                 com[9 - num_point*2] = round(min(max(cp[1] + 15, 0), 15), 3)
        #                 num_point += 1
        # if not comtrans.err:
        #     # print(com)
        #     com_out = procotol(com)
        #     # print(com_out)
        #     comtrans.send_data(com_out)
        # resized = cv2.resize(cam_distort,(960,640))
        # cv2.imshow('result',resized)
    #except Exception as e:
    #    print("main camera error:",e)
    #    print("session end.")
    #    Camera.stop()

if __name__ == "__main__":
    args = parse_args()
    if args.color == 0:
        IS_RED = True
    else:
        IS_RED = False
    _,rvec,tvec = locate_record(int(not IS_RED),save=False)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.reshape(-1)
    T = np.linalg.inv(T)
    record = args.record # 是否录制,默认为0(不录制)
    rospy.init_node('listener', anonymous=True)
    subcap= SubCamera()
    cap=Camera()
    cap.start()
    subcap.start()
    lidar = Radar(K_0, C_0, E_0, imgsz=imgsz)
    sublidar = Radar(K_1, C_1, E_1, imgsz=imgsz1)
    lidar.start()
    sublidar.start()
    bridge = CvBridge()
    while True:
        try:
            main_camera_gen(cap, bridge, lidar,sublidar, args,subcap)
        except:
            pass
        if EXIT_FLAG:
            print("EXIT")
            break
    # comtrans.close()


