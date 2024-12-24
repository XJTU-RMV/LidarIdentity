#!/usr/bin/env python3
#!coding=utf-8

import serial
import serial.tools.list_ports
import threading
import struct
from time import sleep

BPS=115200
TIMEX=5

class ComTrans(object):
    def __init__(self, port, bps=BPS, timex=TIMEX):
        self.err = False
        self.run_status = 0
        try:
            self.com = serial.Serial(port, bps, timex)
            self.run_status = 1
            print('[com] ', self.com.port)
            print('[bps] ',self.com.baudrate)
        except Exception as e:
            print("start error", e)
            self.err = True

    def recv_thread(self):
        try:
            data = self.com.readline()
            data = "[com==>pc] " + data.decode()
            print(data)
            sleep(0.05)
        except Exception as e:
            print("[recv error] ", e)

    def recv_start(self):
        print("start recv_thread")
        threading.Thread(target=self.recv_thread, daemon=True).start()
        
    def close(self):
        print("close com")
        self.com.close()
        
    def send_data(self, data):
        #print("[pc==>uart] ", data)
        self.com.write(data)

def procotol(input_list):
    #input = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0]
    output_str=b''
    for i in input_list:
        pre = struct.pack('<f',i)
        output_str+=pre
    return output_str

if __name__ == '__main__':
    port_list = list(serial.tools.list_ports.comports())
    print(port_list)
    if len(port_list) == 0:
        print("no coms")
    else:
        for i in range(0, len(port_list)):
            print(port_list[i])
    o = procotol([18.235, 6.782, 12.123, 13.782, 0, 0, 4.523, 12.123, 24.758, 8.444])
    comtrans = ComTrans("/dev/ttyUSB0")
    
    if not comtrans.err:
        epoch = 1000
        while(True):
            print(len(o))
            comtrans.send_data(o)
            sleep(0.05)
            epoch -= 1
        comtrans.close()
    