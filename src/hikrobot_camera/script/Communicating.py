# -*- encoding=utf-8 -*-
import serial
import time
import threading
import pdb
import struct
import mmap
import contextlib
from datetime import datetime
import time
import multiprocessing

CRC8_TAB = [0x00, 0x5e, 0xbc, 0xe2, 0x61, 0x3f, 0xdd, 0x83, 0xc2, 0x9c, 0x7e, 0x20, 0xa3, 0xfd, 0x1f, 0x41,
            0x9d, 0xc3, 0x21, 0x7f, 0xfc, 0xa2, 0x40, 0x1e, 0x5f, 0x01, 0xe3, 0xbd, 0x3e, 0x60, 0x82, 0xdc,
            0x23, 0x7d, 0x9f, 0xc1, 0x42, 0x1c, 0xfe, 0xa0, 0xe1, 0xbf, 0x5d, 0x03, 0x80, 0xde, 0x3c, 0x62,
            0xbe, 0xe0, 0x02, 0x5c, 0xdf, 0x81, 0x63, 0x3d, 0x7c, 0x22, 0xc0, 0x9e, 0x1d, 0x43, 0xa1, 0xff,
            0x46, 0x18, 0xfa, 0xa4, 0x27, 0x79, 0x9b, 0xc5, 0x84, 0xda, 0x38, 0x66, 0xe5, 0xbb, 0x59, 0x07,
            0xdb, 0x85, 0x67, 0x39, 0xba, 0xe4, 0x06, 0x58, 0x19, 0x47, 0xa5, 0xfb, 0x78, 0x26, 0xc4, 0x9a,
            0x65, 0x3b, 0xd9, 0x87, 0x04, 0x5a, 0xb8, 0xe6, 0xa7, 0xf9, 0x1b, 0x45, 0xc6, 0x98, 0x7a, 0x24,
            0xf8, 0xa6, 0x44, 0x1a, 0x99, 0xc7, 0x25, 0x7b, 0x3a, 0x64, 0x86, 0xd8, 0x5b, 0x05, 0xe7, 0xb9,
            0x8c, 0xd2, 0x30, 0x6e, 0xed, 0xb3, 0x51, 0x0f, 0x4e, 0x10, 0xf2, 0xac, 0x2f, 0x71, 0x93, 0xcd,
            0x11, 0x4f, 0xad, 0xf3, 0x70, 0x2e, 0xcc, 0x92, 0xd3, 0x8d, 0x6f, 0x31, 0xb2, 0xec, 0x0e, 0x50,
            0xaf, 0xf1, 0x13, 0x4d, 0xce, 0x90, 0x72, 0x2c, 0x6d, 0x33, 0xd1, 0x8f, 0x0c, 0x52, 0xb0, 0xee,
            0x32, 0x6c, 0x8e, 0xd0, 0x53, 0x0d, 0xef, 0xb1, 0xf0, 0xae, 0x4c, 0x12, 0x91, 0xcf, 0x2d, 0x73,
            0xca, 0x94, 0x76, 0x28, 0xab, 0xf5, 0x17, 0x49, 0x08, 0x56, 0xb4, 0xea, 0x69, 0x37, 0xd5, 0x8b,
            0x57, 0x09, 0xeb, 0xb5, 0x36, 0x68, 0x8a, 0xd4, 0x95, 0xcb, 0x29, 0x77, 0xf4, 0xaa, 0x48, 0x16,
            0xe9, 0xb7, 0x55, 0x0b, 0x88, 0xd6, 0x34, 0x6a, 0x2b, 0x75, 0x97, 0xc9, 0x4a, 0x14, 0xf6, 0xa8,
            0x74, 0x2a, 0xc8, 0x96, 0x15, 0x4b, 0xa9, 0xf7, 0xb6, 0xe8, 0x0a, 0x54, 0xd7, 0x89, 0x6b, 0x35]
wCRC_Table = [0x0000, 0x1189, 0x2312, 0x329b, 0x4624, 0x57ad, 0x6536, 0x74bf,0x8c48, 0x9dc1, 0xaf5a, 0xbed3, 0xca6c, 0xdbe5, 0xe97e, 0xf8f7,
              0x1081, 0x0108, 0x3393, 0x221a, 0x56a5, 0x472c, 0x75b7, 0x643e,0x9cc9, 0x8d40, 0xbfdb, 0xae52, 0xdaed, 0xcb64, 0xf9ff, 0xe876, 
              0x2102, 0x308b, 0x0210, 0x1399, 0x6726, 0x76af, 0x4434, 0x55bd,0xad4a, 0xbcc3, 0x8e58, 0x9fd1, 0xeb6e, 0xfae7, 0xc87c, 0xd9f5,
              0x3183, 0x200a, 0x1291, 0x0318, 0x77a7, 0x662e, 0x54b5, 0x453c,0xbdcb, 0xac42, 0x9ed9, 0x8f50, 0xfbef, 0xea66, 0xd8fd, 0xc974,
              0x4204, 0x538d, 0x6116, 0x709f, 0x0420, 0x15a9, 0x2732, 0x36bb,0xce4c, 0xdfc5, 0xed5e, 0xfcd7, 0x8868, 0x99e1, 0xab7a, 0xbaf3,
              0x5285, 0x430c, 0x7197, 0x601e, 0x14a1, 0x0528, 0x37b3, 0x263a,0xdecd, 0xcf44, 0xfddf, 0xec56, 0x98e9, 0x8960, 0xbbfb, 0xaa72,
              0x6306, 0x728f, 0x4014, 0x519d, 0x2522, 0x34ab, 0x0630, 0x17b9,0xef4e, 0xfec7, 0xcc5c, 0xddd5, 0xa96a, 0xb8e3, 0x8a78, 0x9bf1,
              0x7387, 0x620e, 0x5095, 0x411c, 0x35a3, 0x242a, 0x16b1, 0x0738,0xffcf, 0xee46, 0xdcdd, 0xcd54, 0xb9eb, 0xa862, 0x9af9, 0x8b70,
              0x8408, 0x9581, 0xa71a, 0xb693, 0xc22c, 0xd3a5, 0xe13e, 0xf0b7,0x0840, 0x19c9, 0x2b52, 0x3adb, 0x4e64, 0x5fed, 0x6d76, 0x7cff,
              0x9489, 0x8500, 0xb79b, 0xa612, 0xd2ad, 0xc324, 0xf1bf, 0xe036,0x18c1, 0x0948, 0x3bd3, 0x2a5a, 0x5ee5, 0x4f6c, 0x7df7, 0x6c7e,
              0xa50a, 0xb483, 0x8618, 0x9791, 0xe32e, 0xf2a7, 0xc03c, 0xd1b5,0x2942, 0x38cb, 0x0a50, 0x1bd9, 0x6f66, 0x7eef, 0x4c74, 0x5dfd,
              0xb58b, 0xa402, 0x9699, 0x8710, 0xf3af, 0xe226, 0xd0bd, 0xc134,0x39c3, 0x284a, 0x1ad1, 0x0b58, 0x7fe7, 0x6e6e, 0x5cf5, 0x4d7c,
              0xc60c, 0xd785, 0xe51e, 0xf497, 0x8028, 0x91a1, 0xa33a, 0xb2b3,0x4a44, 0x5bcd, 0x6956, 0x78df, 0x0c60, 0x1de9, 0x2f72, 0x3efb,
              0xd68d, 0xc704, 0xf59f, 0xe416, 0x90a9, 0x8120, 0xb3bb, 0xa232,0x5ac5, 0x4b4c, 0x79d7, 0x685e, 0x1ce1, 0x0d68, 0x3ff3, 0x2e7a,
              0xe70e, 0xf687, 0xc41c, 0xd595, 0xa12a, 0xb0a3, 0x8238, 0x93b1,0x6b46, 0x7acf, 0x4854, 0x59dd, 0x2d62, 0x3ceb, 0x0e70, 0x1ff9,
              0xf78f, 0xe606, 0xd49d, 0xc514, 0xb1ab, 0xa022, 0x92b9, 0x8330,0x7bc7, 0x6a4e, 0x58d5, 0x495c, 0x3de3, 0x2c6a, 0x1ef1, 0x0f78]

# def float2byte(f):
#     return [hex(i) for i in struct.pack('f', f)]

class Port:
    #设置端口的状态
    def __init__(self, port, band): 
        self.port = port                    #设定端口号名称
        self.baud = int(band)               #端口传输速度设定为整数（手册中指定为115200）
        self.__open_port = None             #标识串口是否打开（bool类型）
        self.get_data_flag = True           #控制开始或停止接收数据（bool类型）
        self.__real_time_all_data = b''     #接受的所有数据
        self.__real_time_one_data = b''     #一次接受的数据

    #返回已接收的全部数据
    def get_real_time_all_data(self): 
        ret = self.__real_time_all_data     #接收的数据存入ret
        self.__real_time_all_data = b''     #清空接收的数据
        return ret    
    #返回一次接收的数据                      
    def get_real_time_one_data(self): 
        return self.__real_time_one_data
    #清空所有已接收的数据
    def clear_real_time_data(self): 
        self.__real_time_all_data = b''
    #传递bool值，设置是否接收数据
    def set_get_data_flag(self, get_data_flag): 
        self.get_data_flag = get_data_flag

    #打开串口通信
    def openPort(self):
        try:
            self.__open_port = serial.Serial(self.port, self.baud)      #serial.Serial函数打开串口连接
            threading.Thread(target=self.get_data, args=()).start()     #如果串口成功打开，创建一个新的线程，get_data 方法将在后台线程中运行，而不会阻塞主线程的执行
            print("Open port successfully")
        except Exception as e:
            print('Open com fail:{}/{}'.format(self.port, self.baud))   #打开失败的处置
            print('Exception:{}'.format(e))

    #关闭串口通信
    def close(self):
        if self.__open_port is not None and self.__open_port.isOpen:    #检查 self.__open_port 是否为 None 且串口是否处于打开状态
            self.get_data_flag = False                                  
            self.__open_port.close()

    #向串口发送数据
    def send_data(self, data):
        #如果串口是关闭的就打开它
        if self.__open_port is None:
            self.openPort()
        #再次检查
        if self.__open_port is None:
            print("Warring: Port is None!")
            return 0
        #如果打开则写入发送的数据
        if self.__open_port.isOpen():
            success_bytes = self.__open_port.write(data)
            # print("Successfully writen!")
            return success_bytes
        else:
            print("Warring: Port is closed!")

    #从串口接收数据
    def get_data(self):
        #如果串口是关闭的就打开它
        if self.__open_port is None:
            self.openPort()
        while self.__open_port.isOpen():    #只要串口保持打开状态就持续执行
            if self.get_data_flag:          #get_data_flag为true时才会接受数据
                n = self.__open_port.inWaiting()    #检查串口输入缓冲区中是否有待读取的数据，如果有，则n>0
                if n:                               #如果n>0说明有待读取的数据
                    data = self.__open_port.read()  #一次读所有的字节
                    # print('当前向串口传输的数据为：',data)         #显示
                    self.__real_time_all_data += data   #最新接收到的数据也存储在self.__real_time_one_data中
                    self.__real_time_one_data = data    #存储最后一次接收到的数据
                    # print(self.__real_time_all_data)
        print('向串口传输的数据为：',self.__real_time_all_data)
        print("Warning: Port is closed!")

uint8_t = 'B'   #0~255的整数被格式化为      单字节 的二进制字符串
uint16_t = 'H'  #0~65535的整数被格式化为    双字节 的二进制字符串
uint32_t = 'I'  #0~(2^32-1)的整数被格式化为 四字节 的二进制字符串

	 

#接口协议说明请参考《裁判系统学生串口协议附录》,在python中，数据流以bytes类型处理
class Communicator:
    #定义该类的属性
    def __init__(self):
        """

        Returns
        -------
        object
        """
        self.header_SOF = b'\xa5'       #字节串，用作数据包的起始标志
        self.CRC8_INIT = b'\xff'        #用于 CRC 校验的初始值
        self.CRC16_INIT = b'\xff\xff'   #用于 CRC 校验的初始值
        self.header_length = 5          #数据包头部的长度
        self.storage_mode = 'little'    #数据的存储模式（小端）
                                        #小端模式（Little Endian）：最低有效字节在前，最高有效字节在后
                                        #大端模式（Big Endian）：最高有效字节在前，最低有效字节在后
        self.Alig_format = '>'          #数据的字节对齐方式
        self.isCRC = True      #数据接收时是否校验
        
    def set_isCRC(self, BOOL):
        self.is_CRC = BOOL


    def package_data(self, cmd_id, seq, data):   #封装数据包，最终的数据包
        header_data_lenth = len(data).to_bytes(2,self.storage_mode)
        header_seq = int(seq).to_bytes(1,self.storage_mode)
        frame_header = self.header_SOF + header_data_lenth + header_seq + self.CRC8_INIT
        frame_header = self.append_CRC8_check_sum(frame_header, len(frame_header))  
        if isinstance(cmd_id,int):
            cmd_id = cmd_id.to_bytes(2, self.storage_mode)
        if(frame_header is not None):
            #将 frame_header、cmd_id 和数据 data 拼接在一起，再加上 CRC16初始值 self.CRC16_INIT
            package = frame_header + cmd_id +data + self.CRC16_INIT
            package = self.append_CRC16_check_sum(package, len(package))
            if(package is not None):    
                return package
        return None
    
    def unpackage_data(self, data_bytes):   #将数据流拆解为数据包
        data_packs = []
        n = len(data_bytes)
        for i in range(n):
            byte = data_bytes[i].to_bytes(1,self.storage_mode)
            if byte == self.header_SOF:
                # print('帧头为',byte)
                # m=data_bytes[i+1].to_bytes(1,self.storage_mode)
                # print(m)
                if i+5 < n:
                    # cmd_id=data_bytes[i+5].to_bytes(2,self.storage_mode)          
                    header = data_bytes[i:i+5]
                    # cmd_id = data_bytes[i+5:i+7]
                    # print('第一次读',cmd_id)
                    # if cmd_id == self.cmd_id_blood:
                    #     print("血量信息")
                    # print(header)
                    data_length = int.from_bytes(header[1:3],byteorder=self.storage_mode)   #将int类型转换为bytes类型，小端格式
                    # print(data_length)
                    # data=data_bytes[i+7:i+7+data_length]
                    if i+5+2+data_length+2 <= n and ((not self.isCRC) or (self.isCRC and self.verify_CRC8_check_sum(header,len(header)))):
                        data_pack = data_bytes[i:i+5+2+data_length+2]
                        if(not self.isCRC) or (self.isCRC and self.verify_CRC16_check_sum(data_pack,(len(data_pack)))):
                            data_packs.append(data_pack)
                            i += 5+2+data_length+2-1
        return data_packs

            
    #CRC校验
    #crc8 generator polynomial:G(x)=x8+x5+x4+1
    """
    输入： bytes类型的pchMessage，int类型dwLength，bytes类型ucCRC8
    输出：bytes类型ucCRC8
    """

    #计算给定消息的 CRC8校验和
    def get_CRC8_check_sum(self, pchMessage, dwLength, ucCRC8):
        ucIndex = 0
        i = 0
        ucCRC8 = int.from_bytes(ucCRC8,self.storage_mode)
        while i < dwLength:
            ucIndex = ucCRC8^(pchMessage[i])
            ucCRC8 = CRC8_TAB[ucIndex]
            i += 1
        return int(ucCRC8).to_bytes(1,self.storage_mode)
    
    #验证给定消息的 CRC8校验和是否正确
    def verify_CRC8_check_sum(self, pchMessage, dwLength):

        if int.from_bytes(pchMessage,self.storage_mode) == 0 or dwLength <=2:
            return False
        ucExpected = self.get_CRC8_check_sum(pchMessage, dwLength-1, self.CRC8_INIT)
        return ucExpected == pchMessage[dwLength-1:]

    #在给定的消息后面添加 CRC8校验和
    def append_CRC8_check_sum(self, pchMessage, dwLength):
        if int.from_bytes(pchMessage,self.storage_mode) == 0 or dwLength <= 2:
            return None
        ucCRC = self.get_CRC8_check_sum(pchMessage, dwLength-1, self.CRC8_INIT)
        pchMessage = pchMessage[0:dwLength-1]+ucCRC
        return pchMessage
    
    """
    输入： bytes类型的pchMessage，int类型dwLength，bytes类型wCRC
    输出：bytes类型wCRC
    """

    #CRC16的校验，同上
    def get_CRC16_check_sum(self, pchMessage,dwLength,wCRC):
        chData = 0x00
        i = 0
        wCRC = int.from_bytes(wCRC,self.storage_mode)
        if (pchMessage == None):
            return int(0xFFFF).to_bytes(2,self.storage_mode)
        while i<dwLength:
            chData = pchMessage[i]
            wCRC = (wCRC >> 8) ^ wCRC_Table[(wCRC ^ chData) & 0x00ff]
            i += 1
        return int(wCRC).to_bytes(2, self.storage_mode)

    def verify_CRC16_check_sum(self, pchMessage, dwLength):
        wExpected = 0
        if pchMessage == None or dwLength <= 2:
            return False
        wExpected = self.get_CRC16_check_sum(pchMessage, dwLength - 2, self.CRC16_INIT)
        wExpected = int.from_bytes(wExpected,self.storage_mode)
        m2 = pchMessage[dwLength - 2]
        m1 = pchMessage[dwLength - 1]
        return ((wExpected & 0xff) == m2 and ((wExpected >> 8) & 0xff) == m1)

    def append_CRC16_check_sum(self, pchMessage, dwLength):
        if (pchMessage == None) or (dwLength <= 2):
            return None
        wCRC = self.get_CRC16_check_sum(pchMessage, dwLength-2, self.CRC16_INIT)
        
        pchMessage = pchMessage[0:dwLength-2]+wCRC
        return pchMessage
    
    #从共享内存中读取位置信息
    def recieve_position(self):
        shmem=mmap.mmap(0,100,'global_share_memory_position',mmap.ACCESS_READ)  #使用 mmap.mmap 函数创建一个映射对象 shmem，代表了共享内存区域
        s=str(shmem.read(shmem.size()).decode("utf-8"))     #使用 shmem.read 方法读取共享内存的内容，并通过 decode("utf-8") 转换为字符串
        return s
        #vs2012早期版本会有截断符和开始符号，需要提取有用字符串
        # es='\\x00'#字符条件截断，还没有设计开始endstring
        # if s.find(es)==-1:
        #     print(s)
        # else:
        #     sn=s[:s.index(s)]
        #     print('other data')
            
    def read_time(self,data):
        time_left = int.from_bytes(data, 'little')
        return time_left


    def analysis_data_pack(self, data_packs):  # 接收数据包分析  (有待完善)
        # status_seconds = 0  #存储状态时间
        for data_pack in data_packs:
            # cmd_id = int.from_bytes(data_pack[5:7], self.storage_mode)
            # print('cmd_id为',cmd_id)
            cmd_id = int.from_bytes(data_pack[5:7] ,self.storage_mode)
            # print('cmd_id:',cmd_id)
            # print('cmd_id_blood:',self.cmd_id_blood)
            if cmd_id == self.cmd_id_blood:
                print("血量信息")
                data_blood = data_pack[7:39]
                self.read_blood(data_blood)
            # if cmd_id == self.cmd_id_status:    #cmd_id_status见153行
            #     status_time = data_pack[8:9]
            #     status_seconds = int.from_bytes(status_time, 'little')
            #     if status_seconds!=0:
            #         return status_seconds
            # else:
            #     continue
        return None
        
    def lmap_interaction_send(self, pos_data, seq):
        data = b''  # 初始化一个空的字节序列
        
		#pos_data中存储所有的坐标
        for i in range(0, len(pos_data), 2):  
            x = int(pos_data[i])
            y = int(pos_data[i + 1])
            data += struct.pack('<HH', x, y)
        # data += struct.
        # ID = int(robot_ID).to_bytes(2,self.storage_mode)
        # data  += ID                
        cmd_id = int(0x0305).to_bytes(2, self.storage_mode)
        return self.package_data(cmd_id, seq, data)
    
    def double_send(self, seq):
        # if self.result != 0 and self.conut != 0:
        #     send_count = send_count+1
        data_R1 = struct.pack('<HHHB', 0x0121, 9 ,0x8080 , 1) #L=Bule 109 Red 9
        data_B1 = struct.pack('<HHHB', 0x0121, 109 ,0x8080 , 1) #L=Bule 109 Red 9
        data_R2 = struct.pack('<HHHB', 0x0121, 9 ,0x8080 , 2) #L=Bule 109 Red 9
        data_B2 = struct.pack('<HHHB', 0x0121, 109 ,0x8080 , 2) #L=Bule 109 Red 9
        cmd_id = struct.pack('<H', 0x0301)
        self.double1_R_datapack = self.package_data(cmd_id, seq, data_R1)
        self.double1_B_datapack = self.package_data(cmd_id, seq, data_B1)
        self.double2_R_datapack = self.package_data(cmd_id, seq, data_R2)
        self.double2_B_datapack = self.package_data(cmd_id, seq, data_B2)
        return None

if __name__ == '__main__':
    com =Port('/dev/ttyUSB0',115200)    #创建一个 Port 对象 com，指定串口设备路径为 /dev/ttyUSB0(linux系统）COMx(windows系统)，波特率为 115200，然后打开这个端口
    com.openPort()
    com_class = Communicator()  #创建一个 Communicator 对象 com_class
    
    
    pos_data = [100, 100, 200, 200, 300, 300, 400, 400, 500, 500, 600, 600]
    send_interact = com_class.lmap_interaction_send(pos_data, 0)
    
    print('send_interact:',send_interact)
    
    
    #t_move=0.1
    # thread1 = threading.Thread(target=com.send_data(send_interact))    #创建一个线程 thread1，目标函数是 send_position，启动这个线程
    # thread1.start()
    #time.sleep(1)
    
    #进入一个无限循环，在这个循环中，首先调用 com.get_real_time_all_data() 获取裁判系统实时数据，
    #然后将这些数据传递给 com_class.unpackage_data(s) 进行解包，最后调用 com_class.analysis_data_pack(s) 分析数据包
    while True:
        # # com.send_data()
        # s = com.get_real_time_all_data()
        # print('s1:\n',s)
        # print('################################################################')
        # s = com_class.unpackage_data(s)
        # print('s2:\n',s)
        # print('################################################################')

    #     # for i in range(len(s)):
    #     #     print(s[i].hex())
    #     # thread1 = threading.Thread(target=send_position(0.11,send_position(0.11,t_move)))
        
    #     # com.send_data(send_draw)
        # com.send_data(send_interact_new)
        # print('ok1')
        # time.sleep(0.21)
        # print('send_interact is:', send_interact)
        com.send_data(send_interact)
        time.sleep(0.2)
        
        #print(s.hex())
        #com.send_data(s)