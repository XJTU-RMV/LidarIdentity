import serial
import numpy as np
import time
from ser_api import *
def SerSend():

    global pointList
    
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.2)
    time_last = time.time()

    while 1:

        send_list = [
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0]
                ] 
        pointList={101:(5,5),102:(5,5),103:(5,5),104:(5,5),105:(5,5),107:(5,5)}
        for id in pointList:
            if id == -1:
                continue

            if pointList[id] ==None:
                continue

            x,y = pointList[id]

            x = x * 100  
            y = y * 100 

            if np.isnan(x):
                continue
            
            id = int(id)
            ENEMY="BLUE"
            if ENEMY == "BLUE":

                if id == 101:
                    send_list[0] = [x,y]

                if id == 102:
                    send_list[1] = [x,y]

                if id == 103:
                    send_list[2] = [x,y]

                if id == 104:
                    send_list[3] = [x,y]

                if id == 105:
                    send_list[4] = [x,y]

                if id == 107:
                    send_list[5] = [x,y]
                

            else:
            # IF ENEMY == "RED":

                if id == 1 :
                    send_list[0] = [x,y]

                if id == 2 :
                    send_list[1] = [x,y]

                if id == 3 :
                    send_list[2] = [x,y]

                if id == 4 :
                    send_list[3] = [x,y]

                if id == 5 :
                    send_list[4] = [x,y]

                if id == 7 :
                    send_list[5] = [x,y]


        print("send_list:",send_list)

        data = build_data_radar(send_list)
        packet = build_send_packet(data,  [0x03, 0x05])


        usedtime = time.time() - time_last
        if usedtime > 0.2:
            usedtime = 0.199
        sleeptime = 0.2 - usedtime

        print(list(packet),len(packet))
        print(receive_packet(packet,[0x03,0x05],True))
        ser.write(packet)
        # print(ser.in_waiting)
        time_last = time.time()

        # cv2.waitKey(200)
        time.sleep(sleeptime)
SerSend()
