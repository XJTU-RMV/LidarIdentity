#!/usr/bin/env python3
#!coding=utf-8

try:
    import rospy
    from sensor_msgs.msg import Image
    
except:
    print("[ERROR] ROS environment hasn't been successfully loaded.")
import cv2
from cv_bridge import CvBridge
import threading
import ctypes
import inspect
import yaml
import numpy as np

# 安全关闭子线程
def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)

class ImageShit():
    def __init__(self):
        self.img = []
    def push_back(self, data):
        self.img = data


class Camera:

    # the global member of the Camera class
    __init_flag = False # 相机启动标志
    __working_flag = False # 相机接收线程启动标志
    __threading = None # 相机接收子线程

    __lock = threading.Lock() # 线程锁
    __queue = [] # 一个列表，存放相机类各个对象的图像

    def __init__(self):
        '''
        相机处理类，对每个相机应用都要创建一个对象
        '''
        if not Camera.__init_flag:
            # 当雷达还未有一个对象时，初始化接收节点
            # TODO
            Camera.__laser_listener_begin(f"/hikrobot_camera/rgb")
            Camera.__init_flag = True
            Camera.__threading=threading.Thread(target = Camera.__main_loop,daemon=True)
        self._no = len(Camera.__queue) # 该对象对应于整个雷达对象列表的序号
        Camera.__queue.append(ImageShit())
    @staticmethod
    def start():
        '''
        开始子线程，即开始spin
        '''
        if not Camera.__working_flag:
            Camera.__threading.start()
            Camera.__working_flag = True
            print("camera starts.")
    @staticmethod
    def stop():
        '''
        结束子线程
        '''
        if Camera.__working_flag:
            stop_thread(Camera.__threading)
            Camera.__working_flag = False
            print("Camera stops.")

    @staticmethod
    def __callback(data):
        '''
        子线程函数，对于/hikrobot_camera/rgb topic数据的处理
        '''
        #print('Shit')
        if Camera.__working_flag:
            Camera.__lock.acquire()
            # update every class object's queue
            for q in Camera.__queue:
                q.push_back(data)

            Camera.__lock.release()

    @staticmethod
    def __laser_listener_begin(laser_node_name = f"/hikrobot_camera/rgb"):
        # rospy.init_node('webcam_display', anonymous=True)
        rospy.Subscriber(laser_node_name, Image,Camera.__callback)
    @staticmethod
    def __main_loop():
        # 通过将spin放入子线程来防止其对主线程的阻塞
        rospy.spin()
        # 当spin调用时，subscriber就会开始轮询接收所订阅的节点数据，即不断调用callback函数

    def read(self):
        '''
        读取数据
        '''
        flag = False
        Camera.__lock.acquire()
        img = Camera.__queue[self._no].img
        Camera.__lock.release()
        if img != []:
            flag = True
        return flag, img

    def __del__(self):
        Camera.stop()

class SubCamera:
    # the global member of the Camera class
    __init_flag = False # 相机启动标志
    __working_flag = False # 相机接收线程启动标志
    __threading = None # 相机接收子线程

    __lock = threading.Lock() # 线程锁
    __queue = [] # 一个列表，存放相机类各个对象的图像

    def __init__(self):
        '''
        相机处理类，对每个相机应用都要创建一个对象
        '''
        if not SubCamera.__init_flag:
            # 当雷达还未有一个对象时，初始化接收节点
            # TODO
            SubCamera.__laser_listener_begin('/hikrobot_subcamera/rgb')
            SubCamera.__init_flag = True
            SubCamera.__threading=threading.Thread(target = SubCamera.__main_loop,daemon=True)
        self._no = len(SubCamera.__queue) # 该对象对应于整个雷达对象列表的序号
        SubCamera.__queue.append(ImageShit())

    @staticmethod
    def start():
        '''
        开始子线程，即开始spin
        '''
        if not SubCamera.__working_flag:
            SubCamera.__threading.start()
            SubCamera.__working_flag = True
            print("Long camera starts.")
    @staticmethod
    def stop():
        '''
        结束子线程
        '''
        if SubCamera.__working_flag:
            stop_thread(SubCamera.__threading)
            SubCamera.__working_flag = False
            print("Long camera stops.")

    @staticmethod
    def __callback(data):
        '''
        子线程函数，对于/hikrobot_camera/rgb topic数据的处理
        '''
        #print('Shit')
        if SubCamera.__working_flag:
            SubCamera.__lock.acquire()
            # update every class object's queue
            for q in SubCamera.__queue:
                q.push_back(data)

            SubCamera.__lock.release()

    @staticmethod
    def __laser_listener_begin(laser_node_name = "/hikrobot_subcamera/rgb"):
        # rospy.init_node('webcam_display', anonymous=True)
        rospy.Subscriber(laser_node_name, Image,SubCamera.__callback)
    @staticmethod
    def __main_loop():
        # 通过将spin放入子线程来防止其对主线程的阻塞
        rospy.spin()
        # 当spin调用时，subscriber就会开始轮询接收所订阅的节点数据，即不断调用callback函数

    def read(self):
        '''
        读取数据
        '''
        flag = False
        SubCamera.__lock.acquire()
        img = SubCamera.__queue[self._no].img
        SubCamera.__lock.release()
        if img != []:
            flag = True
        return flag, img

    def __del__(self):
        SubCamera.stop()

class ShortCamera:

    # the global member of the Camera class
    __init_flag = False # 相机启动标志
    __working_flag = False # 相机接收线程启动标志
    __threading = None # 相机接收子线程

    __lock = threading.Lock() # 线程锁
    __queue = [] # 一个列表，存放相机类各个对象的图像

    def __init__(self):
        '''
        相机处理类，对每个相机应用都要创建一个对象
        '''
        if not ShortCamera.__init_flag:
            # 当雷达还未有一个对象时，初始化接收节点
            # TODO
            ShortCamera.__laser_listener_begin('/hikrobot_short/rgb')
            ShortCamera.__init_flag = True
            ShortCamera.__threading=threading.Thread(target = ShortCamera.__main_loop,daemon=True)
        self._no = len(ShortCamera.__queue) # 该对象对应于整个雷达对象列表的序号
        ShortCamera.__queue.append(ImageShit())

    @staticmethod
    def start():
        '''
        开始子线程，即开始spin
        '''
        if not ShortCamera.__working_flag:
            ShortCamera.__threading.start()
            ShortCamera.__working_flag = True
            print("short camera starts.")
    @staticmethod
    def stop():
        '''
        结束子线程
        '''
        if ShortCamera.__working_flag:
            stop_thread(ShortCamera.__threading)
            ShortCamera.__working_flag = False
            print("short camera stops.")

    @staticmethod
    def __callback(data):
        '''
        子线程函数，对于/hikrobot_camera/rgb topic数据的处理
        '''
        #print('Shit')
        if ShortCamera.__working_flag:
            ShortCamera.__lock.acquire()
            # update every class object's queue
            for q in ShortCamera.__queue:
                q.push_back(data)

            ShortCamera.__lock.release()

    @staticmethod
    def __laser_listener_begin(laser_node_name = "/hikrobot_short/rgb"):
        # rospy.init_node('webcam_display', anonymous=True)
        rospy.Subscriber(laser_node_name, Image,ShortCamera.__callback)
    @staticmethod
    def __main_loop():
        # 通过将spin放入子线程来防止其对主线程的阻塞
        rospy.spin()
        # 当spin调用时，subscriber就会开始轮询接收所订阅的节点数据，即不断调用callback函数

    def read(self):
        '''
        读取数据
        '''
        flag = False
        ShortCamera.__lock.acquire()
        img = ShortCamera.__queue[self._no].img
        ShortCamera.__lock.release()
        if img != []:
            flag = True
        return flag, img

    def __del__(self):
        ShortCamera.stop()

def read_yaml(camera_type):
    '''
    读取相机标定参数,包含外参，内参，以及关于雷达的外参

    :param camera_type:相机编号
    :return: 读取成功失败标志位，相机内参，畸变系数，和雷达外参，相机图像大小
    '''
    # yaml_path = "./{0}/camera{1}.yaml".format('camerainfo',
    #                                         camera_type)
    # print(yaml_path)
    yaml_path = "/home/qianzezhong/Documents/VSCode_Projects/lidar_new/src/hikrobot_camera/script/camerainfo/camera{0}.yaml".format(camera_type)
    try:
        with open(yaml_path, 'rb')as f:
            res = yaml.load(f, Loader=yaml.FullLoader)
            K_0 = np.float32(res["K_0"]).reshape(3, 3)
            C_0 = np.float32(res["C_0"])
            E_0 = np.float32(res["E_0"]).reshape(4, 4)
            imgsz = tuple(res['ImageSize'])
            print('success read yaml')
        return True, K_0, C_0, E_0, imgsz
    except Exception as e:
        print("[ERROR] {0}".format(e))
        return False, None, None, None, None


if __name__ == '__main__':
    import traceback
    cap = Camera()
    Camera.start()
    bridge = CvBridge()
    #cv2.namedWindow("out",cv2.WINDOW_NORMAL) # 显示相机图像
    try:
        
        key = cv2.waitKey(1)

        while (key != ord('q') & 0xFF):
            
            flag, img = cap.read() # 获得深度图
            if flag:
                #print(img)
                re_img = bridge.imgmsg_to_cv2(img, "bgr8")
                #print(re_img)
                resized_img = cv2.resize(re_img,(960,640))
                cv2.imshow('shit',resized_img)
                key = cv2.waitKey(3)
    except:
        traceback.print_exc()
    Camera.stop()
    cv2.destroyAllWindows()
