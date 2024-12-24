import numpy as np
import torch
from datetime import datetime
import cv2
import time
from scipy.optimize import linear_sum_assignment
from commute import SerialCommunicator
import math
ARMOR_CLASS=['B1','B2','B3','B4','B5','B7',
'R1','R2','R3','R4','R5','R7']
RED_BLIND_POINTS=[[15.5, 7.5-15],[19, 3-15],[21, 6-15],[19, 9.3-15],[13.5, 11-15]] # 红方盲区
BLUE_BLIND_POINTS=[[12.5,-7.5],[9,-3],[7,-6],[9,-9.3],[14.5,-11]] # 蓝方盲区
# BLUE_BLIND_POINTS=[]
def pixel_x(x):
    return 20*x+100
def pixel_y(y):
    return -20*y+400
def get_time():
    seconds_since_epoch = time.time()
    perf_counter = time.perf_counter()
    milliseconds_since_epoch = (seconds_since_epoch + perf_counter) * 1000
    return milliseconds_since_epoch/1000
class tracker:
    def __init__(self,IS_RED):
        self.is_red=IS_RED
        self.cars=[] #里面存放着很多的x,y坐标 都是幽灵车
        self.life_time=[]
        self.now_time=get_time() #单位：秒
        self.live_second=2 #最多存活时间
        self.labels=[]
        self.scores=[]
        self.communitor=SerialCommunicator(IS_RED)
        self.last_datalist=[[1,999,999],[2,999,999],[3,999,999],[4,999,999],[5,999,999],[6,999,999]]
        if self.is_red:
            self.core_points=[[1,22,7.5],[2,22,7.5],[3,22,7.5],[4,22,7.5],[5,22,7.5],[6,22,7.5]]
            self.last_datalist=[[1,22,7.5],[2,22,7.5],[3,22,7.5],[4,22,7.5],[5,22,7.5],[6,22,7.5]]
        else:
            self.core_points=[[1,6,7.5],[2,6,7.5],[3,6,7.5],[4,6,7.5],[5,6,7.5],[6,6,7.5]]
            self.last_datalist=[[1,6,7.5],[2,6,7.5],[3,6,7.5],[4,6,7.5],[5,6,7.5],[6,6,7.5]]
        self.last_score=[0,0,0,0,0,0]
        self.work_series=[[],[],[],[],[],[]]
        self.file=open("./log.txt","w")
    def generate_work_series(self,i):
        x=self.core_points[i][1]
        y=self.core_points[i][2]
        if self.is_red:
            points = RED_BLIND_POINTS
        else:
            points = BLUE_BLIND_POINTS
        
        sorted_points = sorted(points, key=lambda point: math.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2))
        self.work_series[i] = sorted_points
    def filter(self):
        max_scores = {}
        for i, label in enumerate(self.labels):
            if label not in max_scores or self.scores[i] > max_scores[label][0]:
                max_scores[label] = (self.scores[i], i)

        filtered_labels = []
        filtered_scores = []
        filtered_cars = []
        filtered_life_time = []
        for label, (score, index) in max_scores.items():
            filtered_labels.append(label)
            filtered_scores.append(score)
            filtered_cars.append(self.cars[index])
            filtered_life_time.append(self.life_time[index])

        self.labels = filtered_labels
        self.scores = filtered_scores
        self.cars = filtered_cars
        self.life_time = filtered_life_time
        

    def processdata(self,datalist):
        process=self.communitor.receive_data()
        # print("core",self.core_points)
        # process=self.last_score
        # print(process)
        # cv2.waitKey(0)
        if process is None:
            for i in range(len(datalist)):
                if datalist[i][2]==999:
                    ...
                    # datalist[i]=self.last_datalist[i] #延续上一次
                else:
                    self.core_points[i]=datalist[i] #更新失配搜索中心
            return datalist
        else:
            current_time = datetime.now()
            time_str = current_time.strftime("%m-%d %H:%M:%S")
            self.file.write(f"[{time_str}] process:{process} last_data:{self.last_datalist}\n")
            for i in range(len(datalist)):
                
                if datalist[i][2] !=999: #如果雷达扫到了，不做处理
                    self.core_points[i]=datalist[i]
                    continue 
                else: # 使用上一次的,man
                    # datalist[i][1]=self.last_datalist[i][1]
                    # datalist[i][2]=self.last_datalist[i][2]
                    ...
                
                # if process[i] !=0 and(process[i]==120 or process[i]-self.last_score[i]>=0): #上一轮预测是精准的
                #     datalist[i][1]=self.last_datalist[i][1]
                #     datalist[i][2]=self.last_datalist[i][2]
                #     self.core_points[i]=datalist[i]

                # if datalist[i][2]==999:
                #     ... # do nothing. what can I say? 辽科 out
                #     if len(self.work_series[i])==0:
                #         self.generate_work_series(i) # 根据core point生成新的work_series
                #         # print("work_series",self.work_series[i])
                #     datalist[i][1]=self.work_series[i][0][0]
                #     datalist[i][2]=self.work_series[i][0][1]
                #     if len(self.work_series[i])>1:
                #         self.work_series[i]=self.work_series[i][1:]
                #     else:
                #         self.work_series[i]=[]
            ...
            self.last_score=process
            return datalist

    def update(self,cps,armors:list[int,float]):
        now_time=get_time()
        if self.now_time==now_time:
            return
        delta_time=now_time-self.now_time
        self.now_time=now_time
        matched_indices = []
        if len(cps) != 0:
            cps = np.array(cps)
            nan_indices = np.isnan(cps).any(axis=1)
            cps = cps[~nan_indices]
        deleted_idx=[]
        for i in range(len(self.cars)):
            # self.cars[i][0]+=delta_time*self.v[i][0]
            # self.cars[i][1]+=delta_time*self.v[i][1]
            self.life_time[i]-=delta_time
            flag=False
            for j in range(len(self.cars)):
                if self.labels[j]==self.labels[i] and self.scores[i]<self.scores[j]:
                    flag=True
            if self.life_time[i]<=0 or self.scores[i]<0.1 or flag:
                deleted_idx.append(i)
        for idx in reversed(deleted_idx):
            del self.cars[idx]
            del self.life_time[idx]
            # del self.v[idx]
            del self.scores[idx]
            del self.labels[idx]
        if len(self.cars) > 0 and len(cps) > 0:
            cost_matrix = np.zeros((len(cps), len(self.cars)))
            for i, cp in enumerate(cps):
                for j, car in enumerate(self.cars):
                    cost_matrix[i, j] = np.linalg.norm(np.array(cp) - np.array(car))
                    if armors[i][0] !=-1 and self.labels[j] != armors[i][0]:#label不匹配的惩罚项
                        cost_matrix[i, j] +=self.scores[j]*armors[i][1]*3
                    if armors[i][0]==self.labels[j]: # 匹配的奖励项
                        cost_matrix[i, j] -=self.scores[j]*armors[i][1]
                        

            row_ind, col_ind = linear_sum_assignment(cost_matrix) 

            for i, j in zip(row_ind, col_ind):
                distance_threshold = 1.5 #可容忍
                if armors[i][0] !=-1 and self.labels[j] != armors[i][0]:#label不匹配的惩罚项
                    distance_threshold=0
                if cost_matrix[i, j] <= distance_threshold:
                    matched_indices.append((i, j))
        for i in range(len(cps)):
            isMatched=False
            idx=-1
            for j in range(len(matched_indices)):
                if i==matched_indices[j][0]:
                    isMatched=True
                    idx=matched_indices[j][1]
                    break
            if isMatched: #更新幽灵
                self.life_time[idx]=min(self.live_second,self.life_time[idx]+self.live_second*delta_time*3)
                if armors[i][0]==-1:
                    self.scores[idx]*=0.5**delta_time #car匹配到了，结果没识别到装甲版的衰减
                elif self.labels[idx]!=armors[i][0]:
                    self.scores[idx]-=delta_time*((armors[i][1])**2) # 最后常数取决了更换label的速度
                    if self.scores[idx]<min(armors[i][1],0.5):
                        self.labels[idx]=armors[i][0]
                        self.scores[idx]=min(armors[i][1],delta_time*armors[i][1]*2)
                else:
                    self.scores[idx]=min(self.scores[idx]+armors[i][1]*delta_time*2,1)
                # +self.v[idx][0]*delta_time 表示回到原来位置
                # self.v[idx][0]=self.v[idx][0]*(0.5**delta_time)+(1-(0.5**delta_time))*(cps[i][0]-self.cars[idx][0]+self.v[idx][0]*delta_time)/delta_time # 0.5:0.5滤波一下，有待增强
                # self.v[idx][1]=self.v[idx][1]*(0.5**delta_time)+(1-(0.5**delta_time))*(cps[i][1]-self.cars[idx][1]+self.v[idx][1]*delta_time)/delta_time 
                self.cars[idx][0]=cps[i][0]
                self.cars[idx][1]=cps[i][1]
            else: # 添加新车
                if armors[i][0]!=-1:
                    self.cars.append(cps[i])
                    # self.v.append([0,0])
                    self.labels.append(armors[i][0])
                    self.scores.append(min(armors[i][1]*delta_time,armors[i][1]))
                    self.life_time.append(min(0.1+delta_time*3,self.live_second))
        for i in range(len(self.cars)):
            isMatched=False
            idx=-1
            for j in range(len(matched_indices)):
                if i==matched_indices[j][1]:
                    isMatched=True
                    idx=matched_indices[j][0]
                    break
            if not isMatched:
                # self.v[i][0]*=0.5**delta_time
                # self.v[i][1]*=0.5**delta_time
                self.scores[i]*=0.5**delta_time
        #print(len(self.v),len(self.cars),len(self.life_time),len(self.labels),len(self.scores))
        self.filter()
        # print(matched_indices,len(self.cars))
        height, width = 500, 700 
        blue_background = np.zeros((height, width, 3), dtype=np.uint8)
        background_map=cv2.resize(cv2.imread("/home/qianzezhong/Documents/VSCode_Projects/lidar_new/src/hikrobot_camera/script/static/map.png"),(560,300))
        # print(int(pixel_y(0)),int(pixel_y(15)))
        # print(int(pixel_x(0)),int(pixel_x(28)))
        # print(blue_background.shape)
        blue_background[:, :, :] = (255, 255, 255)
        blue_background[int(pixel_y(15)):int(pixel_y(0)),int(pixel_x(0)):int(pixel_x(28)),:]=background_map
        
        data_list=[[1,999,999],[2,999,999],[3,999,999],[4,999,999],[5,999,999],[6,999,999]]
        cv2.line(blue_background,(pixel_x(0),pixel_y(0)),(pixel_x(0),pixel_y(15)),(0,0,0),2)
        cv2.line(blue_background,(pixel_x(0),pixel_y(15)),(pixel_x(28),pixel_y(15)),(0,0,0),2)
        cv2.line(blue_background,(pixel_x(28),pixel_y(15)),(pixel_x(28),pixel_y(0)),(0,0,0),2)
        cv2.line(blue_background,(pixel_x(28),pixel_y(0)),(pixel_x(0),pixel_y(0)),(0,0,0),2)
        for i in range(len(self.cars)):
            if self.scores[i] < 0.5:
                continue
            if self.life_time[i]<0.5:
                continue
            car=self.cars[i]
            x, y = car[0], car[1]
            if -50 <= x < width + 50 and -50 <= y < height + 50:  
                y+=15
                if self.labels[i] <=5:
                    cv2.circle(blue_background, (int(pixel_x(x)), int(pixel_y(y))), 3, (255, 0, 0), -1)
                else:
                    cv2.circle(blue_background, (int(pixel_x(x)), int(pixel_y(y))), 3, (0, 0, 255), -1)
                    
                    
                cv2.putText(blue_background,f"{ARMOR_CLASS[self.labels[i]]}:{x:.2f}:{y:.2f}:{self.life_time[i]:.2f}:{self.scores[i]:.2f}",
                            (int(pixel_x(x)), int(pixel_y(y))),cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, [0,255, 0],
                            thickness=2)
                if self.is_red and self.labels[i]<=5:
                    if self.labels[i]==5:
                        data_list[5][1]=x
                        data_list[5][2]=y
                    else:
                        data_list[self.labels[i]][1]=x
                        data_list[self.labels[i]][2]=y    
                if not self.is_red and self.labels[i]>5:
                    if self.labels[i]==11:
                        data_list[5][1]=x
                        data_list[5][2]=y
                    else:
                        data_list[self.labels[i]-6][1]=x
                        data_list[self.labels[i]-6][2]=y
        _data=self.processdata(data_list)
        self.last_datalist=_data
        # print('sending')
        self.communitor.send_data(data_list=_data)
        # print('done')
        
        cv2.imshow('Tracking', blue_background)
        cv2.waitKey(1)
    def stop(self):
        self.communitor.close()
        self.file.close()
if __name__=="__main__":
    t=tracker(True)
    t.update([[1,-2],[3,-4]],[[1,0.9],[2,0.9]])
    time.sleep(0.1)
    t.update([[1,-2],[3,-4]],[[1,0.9],[2,0.9]])
    time.sleep(0.1)
    t.update([[1,2],[3,4]],[[1,0.9],[2,0.9]])
    time.sleep(0.1)
    t.update([[1,-2],[3,-4]],[[1,0.9],[2,0.9]])
    time.sleep(0.1)
    t.update([[1,-2],[3,-4]],[[1,0.9],[2,0.9]])
    time.sleep(0.1)
    t.update([],[])
    time.sleep(1)
    t.update([],[])
    time.sleep(1)
    t.update([],[])
    time.sleep(1)
    t.update([],[])
    time.sleep(1)
    t.update([],[])
    time.sleep(1)
    t.update([],[])
    time.sleep(1)
    t.update([],[])
    time.sleep(1)
    t.update([],[])
    time.sleep(1)
