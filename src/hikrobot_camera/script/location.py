#!/usr/bin/env python3
#!coding=utf-8

'''
位姿估计函数
进行手动位姿估计
'''
from itertools import cycle
import cv2
import numpy as np
import os
from datetime import datetime
from camera import read_yaml,Camera
import rospy
from cv_bridge import CvBridge
from argparse import ArgumentParser
def __callback_1(event,x,y,flags,param):
    '''
    鼠标回调函数
    鼠标点击点：确认标定点并在图像上显示
    鼠标位置：用来生成放大图
    '''
    # using EPS and MAX_ITER combine
    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                         30, 0.001)
    if event == cv2.EVENT_MOUSEMOVE:
        # 周围200*200像素放大图
        rect = cv2.getWindowImageRect(param["pick_winname"])
        img_cut = np.zeros((200,200,3),np.uint8)
        img_cut[max(-y+100,0):min(param["pick_img"].shape[0]+100-y,200),max(-x+100,0):min(param["pick_img"].shape[1]+100-x,200)] = \
        param["pick_img"][max(y-100,0):min(y+100,param["pick_img"].shape[0]),max(x-100,0):min(x+100,param["pick_img"].shape[1])]
        cv2.circle(img_cut,(100,100),1,(0,255,0),1)
        cv2.imshow(param["zoom_winname"], img_cut)
        cv2.moveWindow(param["zoom_winname"],rect[0]-400,rect[1]+200)
        cv2.resizeWindow(param["zoom_winname"], 400,400)
    if event == cv2.EVENT_LBUTTONDOWN and not param["pick_flag"]:
        param["pick_flag"] = True # 在这里捕获点击
        print(f"pick ({x:d},{y:d})")
        # 亚像素精确化
        corner = cv2.cornerSubPix(param["pick_img_raw"],np.float32([x,y]).reshape(1,1,2),(5,5),(-1,-1),stop_criteria).reshape(2)
        param["pick_point"] = [corner[0],corner[1]]
        cv2.circle(param["pick_img"],(x,y),2,(0,255,0),1)

def __callback_2(event,x,y,flags,param):
    print("callback")
    if event == cv2.EVENT_LBUTTONDOWN and not param:
        param = True
def compute_reprojection_error(ops, pick_point, K_0, C_0, rvec, tvec):
    # 重投影物体点到图像上
    projected_points, _ = cv2.projectPoints(ops, rvec, tvec, K_0, C_0)
    print(projected_points,pick_point,projected_points - pick_point)
    # 计算每个点的重投影误差
    reprojection_errors = np.sqrt(np.sum((projected_points - pick_point)**2, axis=1))
    
    # 返回平均重投影误差
    return np.mean(reprojection_errors)
def locate_pick(cap:Camera,color,bridge,video_test=False):
    '''
    手动四点标定

    :param cap:Camera_Thread object
    :param enemy:enemy color
    :param camera_type:camera number
    :param home_size: 选用在家里测试时的尺寸
    :param video_test: 是否用视频测试，以减慢播放速度

    :return: 读取成功标志，旋转向量，平移向量
    '''
    
    K_0, C_0= read_yaml(0)[1:3]
    P = K_0
    K = C_0[0:4]
    # frame_copy = frame.copy()
    # 窗口下方提示标定哪个目标
    # 己方为红方：4,1,6,5顺序;为蓝方：7,9,5,6
    tips = \
    {
        '0':['centerZhedian','QianshaozhanZhedian','DuimianQSZGreen','QianshaozhanDing','BaseGreen','R_sign'],
        '1':['centerZhedian','QianshaozhanZhedian','DuimianQSZGreen','QianshaozhanDing','BaseGreen','R_sign'],
    }

    backup_coordinate={
        'R4':[5.23,-12.81,0.4],
        'R3':[3.085,-2.11,0.4],
        'R0':[8.67,-5.715,0.42],
        'R1':[7.77,-0.21,0.15],
        'R2':[11.07,-3.47,0.6],
        'B2':[16.77,-11.11,0.6],
        'B1':[19.83,-14.49,0.15],
        'B0':[19.33,-9.285,0.42],
        'B3':[24.505,-12.59,0.4],
        'B4':[22.37,-1.89,0.4]

    }
    # red_coordinate = {
    #     'R1':[9.715,-0.42,0.2],
    #     'R4':[4.825,-12.77,0.4],
    #     'B2':[16.67,-11.505,0.6],
    #     'R2':[11.175,-3.02,0.6]
    # }
    PointCentre = [28000-6517, 15000-7500 ,0] # 中心折点
    PointOutPostPoint = [28000-10078, 15000-2402,0] # 前哨战塔下折点
    PointFenceTop = [28000-28000, 15000-15000,2400] # 栏杆点
    PointOutPost = [28000-11071, 15000-2435 ,1625] # 前哨战顶
    red_coordinate = { # 2024赛季船新版本
        'centerZhedian':[6.5042, 7.500-15 ,0], 
        'QianshaozhanZhedian':[10.014, 2.3736-15,0],
        'DuimianQSZGreen':[16.617,-2.3736,1.449],
        'QianshaozhanDing':[11.071,2.435-15 ,1.625],
        'BaseGreen':[26.040,7.5-15,1.243],
        'R_sign':[13.78,7.18-15,2.394],
        
    }
    # blue_coordinate = {
    #     'B1':[17.885,-14.28,0.2],
    #     'B3':[22.775,-12.77,0.4],
    #     'R2':[11.175,-3.02,0.6],
    #     'B2':[16.67,-11.505,0.6]
    # }
    # B1 B0 B2
    blue_coordinate = { # 2024赛季船新版本
        'centerZhedian':[28-6.5042, -7.500 ,0], 
        'QianshaozhanZhedian':[28-10.014, -2.3736,0],
        'DuimianQSZGreen':[28-16.617,2.3736-15,1.449],
        'QianshaozhanDing':[28-11.071,-2.435 ,1.625],
        'BaseGreen':[28-26.040,-7.5,1.243],
        'R_sign':[28-13.78,-7.18,2.394],
    }
        # OpenCV窗口参数
    info = {}
    
    info["pick_winname"] = "pick_corner"
    info["zoom_winname"] = "zoom_in"
    info["pick_flag"] = False
    info["pick_point"] = None # 回调函数中点击的点
    stop_signal = False
    cv2.namedWindow(info["pick_winname"], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(info["pick_winname"], 1280,780)
    cv2.setWindowProperty(info["pick_winname"],cv2.WND_PROP_TOPMOST,1)
    cv2.moveWindow(info["pick_winname"], 500,300)
    cv2.namedWindow(info["zoom_winname"], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(info["zoom_winname"], 400, 400)
    cv2.setWindowProperty(info["zoom_winname"],cv2.WND_PROP_TOPMOST,1)

    pick_point = []
    # TODO
    # 设定世界坐标
    if color == 0:  # our color is red
        ops = np.float64([i[1] for i in red_coordinate.items()])
    else: # our color is blue
        ops = np.float64([i[1] for i in blue_coordinate.items()])
    ops = ops.reshape(len(red_coordinate),1,3)
    r, frame = cap.read()
    #TODO:修改场地坐标
    # 标定目标提示位置
    Camera.start()
    flag, frame = cap.read()
    while (flag != True):
        flag, frame = cap.read()
    while True:
        # print("loop")
        flag,frame = cap.read()
        re_frame = bridge.imgmsg_to_cv2(frame, "bgr8")
        # re_frame = cv2.imread("/home/qianzezhong/Pictures/scene.png").resize(())
        re_frame = cv2.undistort(re_frame,np.array(P),np.array(K)) # qzz changed
        
        # re_frame=cv2.resize(cv2.imread("/home/qianzezhong/Pictures/scene.png"),(re_frame.shape[1],re_frame.shape[0]))
        cv2.imshow(info["pick_winname"],re_frame)
        # print(stop_signal)
        key = cv2.waitKey(1)
        if key == ord(" "):
            break
            
    cv2.setMouseCallback("pick_corner", __callback_1, info)
    # re_frame=cv2.imread("/home/qianzezhong/Pictures/scene.png")
    # re_frame=cv2.resize(re_frame,(3072, 2048))
    info["pick_img_raw"] = cv2.cvtColor(re_frame, cv2.COLOR_BGR2GRAY)
    info["pick_img"] = re_frame
    tip_w = re_frame.shape[1]//2
    tip_h = re_frame.shape[0]-200

    #print(info)
    while True:
        cv2.imshow(info["pick_winname"], info["pick_img"])
        # draw tips
        frame = re_frame.copy()
        cv2.putText(frame,tips[str(color)][len(pick_point)],(tip_w,tip_h),
                    cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),2)

        # draw the points having been picked
        for select_p in pick_point:
            cv2.circle(frame, (int(select_p[0]), int(select_p[1])), 1, (0, 255, 0), 2)

        # draw the connecting line following the picking order
        for p_index in range(1, len(pick_point)):
            cv2.line(frame, (int(pick_point[p_index - 1][0]), int(pick_point[p_index - 1][1])),
                     (int(pick_point[p_index][0]), int(pick_point[p_index][1])), (0, 255, 0), 2)
        # print(frame)
        cv2.imshow(info["pick_winname"], info["pick_img"])
        if info["pick_flag"]: # 当在回调函数中触发点击事件
            pick_point.append(info["pick_point"])
            # draw the points having been picked
            for select_p in pick_point:
                cv2.circle(frame, (int(select_p[0]), int(select_p[1])), 1, (0, 255, 0), 2)

            # draw the connecting line following the picking order
            for p_index in range(1, len(pick_point)):
                cv2.line(frame, (int(pick_point[p_index - 1][0]), int(pick_point[p_index - 1][1])),
                         (int(pick_point[p_index][0]), int(pick_point[p_index][1])), (0, 255, 0), 2)
            # 四点完成，首尾相连
            if len(pick_point) == len(red_coordinate):
                cv2.line(frame, (int(pick_point[3][0]), int(pick_point[3][1])),
                         (int(pick_point[0][0]), int(pick_point[0][1])), (0, 255, 0), 2)
            cv2.imshow(info["pick_winname"], info["pick_img"])
            # 将刚加入的pop出等待确认后再加入
            pick_point.pop()
            key = cv2.waitKey(0)
            if key == ord('c') & 0xFF: # 确认点加入
                pick_point.append(info["pick_point"])

                print(f"You have pick {len(pick_point):d} point.")

            if key == ord('z') & 0xFF: # 将上一次加入的点也删除（这次的也不要）
                if len(pick_point):
                    pick_point.pop()
                print("drop last")

            if key == ord('q') & 0xFF: # 直接退出标定，比如你来不及了
                cv2.destroyWindow(info["pick_winname"])
                cv2.destroyWindow(info["zoom_winname"])
                return False,None,None
            info["pick_flag"] = False
        else:
            # 当未点击时，持续输出视频
            if video_test:
                cv2.waitKey(80)
            else:
                cv2.waitKey(1)
        if len(pick_point) == len(red_coordinate):  # 四点全部选定完成，进行PNP
            break
        # frame = frame_copy.copy()
        # r, frame = cap.read()
        # if not cap.is_open():
        #     cv2.destroyWindow(info["pick_winname"])
        #     cv2.destroyWindow(info["zoom_winname"])
        #     return False,None,None
        info["pick_img"] = frame

    pick_point = np.float64(pick_point).reshape(-1,1, 2)
    # print(ops,pick_point)
    flag, rvec, tvec = cv2.solvePnP(ops, pick_point, K_0, C_0*0,flags = cv2.SOLVEPNP_ITERATIVE)
    print(compute_reprojection_error(ops,pick_point,K_0,C_0*0,rvec,tvec))
    cv2.waitKey(0)
    cv2.destroyWindow(info["pick_winname"])
    cv2.destroyWindow(info["zoom_winname"])
    return flag,rvec, tvec
from show_process_map import process_map
from test_location import build_lookup_table
def locate_record(enemy,save = False,rvec = None,tvec = None):
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
        build_lookup_table(0,enemy)
        build_lookup_table(1,enemy)
    elif max_order > -1:
        # 读取模型，若文件不为空
        print("####读取模型阶段####")
        flag = True
        print(os.path.join(LOCATION_SAVE_DIR,max_file))
        pose = np.loadtxt(os.path.join(LOCATION_SAVE_DIR,max_file),delimiter=' ').reshape(2,3)
        rvec = pose[0]
        tvec = pose[1]

    return flag,rvec,tvec

if __name__ == "__main__":
    # 创建节点
    rospy.init_node('four_point_listener', anonymous=True)
    cap = Camera()
    # Camera.start()
    bridge = CvBridge()
    parser = ArgumentParser()
    parser.add_argument('--color','-c',type=int)
    args = parser.parse_args()
    flag,rvec,tvec = locate_pick(cap,args.color,bridge,video_test=True)
    if flag:
        
        locate_record(args.color,save=True,rvec=rvec,tvec=tvec)