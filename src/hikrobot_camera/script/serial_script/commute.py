import struct
import serial
import serial.tools
import serial.tools.list_ports
import crcmod
import datetime
from crc_test import Referee
def float_to_bytes(f):
    return struct.pack('f', f)
def uint_to_bytes(u):
    return struct.pack('H', u)
def calculate_crc(byte_data):
    crc8_maxim = crcmod.predefined.Crc('crc-8-maxim')
    crc8_maxim.update(bytes(byte_data))
    return crc8_maxim.crcValue
import datetime
class SerialCommunicator:
    def __init__(self,is_red):
        self.init_time=datetime.datetime.now().timestamp()
        print(self.init_time)
        available_ports = serial.tools.list_ports.comports()
        self.data_list=[
            [1,999,999],
            [2,999,999],
            [3,999,999],
            [4,999,999],
            [5,999,999],
            [6,999,999],
        ]
        if available_ports:
            self.ser = serial.Serial(available_ports[0].device, 115200, timeout=None)
            print(f"Using serial port: {self.ser.port}")
        else:
            print("No serial ports available.")
            self.ser = None
        self.ref=Referee(self.ser,is_red)
        self.ref.run()
    def send_data(self, data_list):
        self.ref.update_datalist(data_list)
    def receive_data(self):
        return None
    def close(self):
        if self.ser:
            self.ser.close()
            print("Serial port closed.")
            self.ref.close()
import time
import random
if __name__ == "__main__":
    communicator = SerialCommunicator(False)
    if True or communicator.ser:
        time.sleep(0.001)
        data_list = [
            [1,1,1],#0~15
            [2,2,2],
            [3,3,3],
            [4,4,4],
            [5,5,5],
            [6,6,6],
        ]
        while True:
            # data=communicator.receive_data()
            communicator.send_data(data_list)
            time.sleep(0.1)
            
        communicator.close()