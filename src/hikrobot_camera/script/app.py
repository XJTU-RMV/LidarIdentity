import tkinter as tk
from tkinter import messagebox
import os
import subprocess

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("ROS Control Panel")

        self.color = tk.StringVar(value="red")

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="选择颜色:").grid(row=0, column=0, padx=10, pady=10)
        
        tk.Radiobutton(self.root, text="红色", variable=self.color, value="red").grid(row=0, column=1)
        tk.Radiobutton(self.root, text="蓝色", variable=self.color, value="blue").grid(row=0, column=2)

        # tk.Button(self.root, text="打开所有ROS节点", command=self.open_ros_nodes).grid(row=1, column=0, columnspan=3, pady=10)
        tk.Button(self.root, text="标定", command=self.calibrate).grid(row=2, column=0, columnspan=3, pady=10)
        tk.Button(self.root, text="启动", command=self.launch).grid(row=3, column=0, columnspan=3, pady=10)
    """
    roslaunch livox_ros_driver livox_lidar.launch
    roslaunch hikrobot_camera hikrobot_camera.launch
    roslaunch hikrobot_camera hikrobot_subcamera.launch
    """
    def open_ros_nodes(self):
        commands = [
            "roslaunch livox_ros_driver livox_lidar.launch",
            "roslaunch hikrobot_camera hikrobot_camera.launch",
            "roslaunch hikrobot_camera hikrobot_subcamera.launch"
        ]
        try:
            for cmd in commands:
                subprocess.Popen(['gnome-terminal', '--tab', '--', 'bash', '-c', f'{cmd}; exec bash'])
                # Adding a sleep to ensure commands don't overlap
                # subprocess.Popen(['sleep', '10']).wait()
            messagebox.showinfo("Success", "所有ROS节点已打开")
        except Exception as e:
            messagebox.showerror("Error", f"打开ROS节点时出错: {e}")
    def calibrate(self):
        color_value = "0" if self.color.get() == "red" else "1"
        try:
            subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', f'/home/qianzezhong/miniconda3/envs/yolov8/bin/python /home/qianzezhong/Documents/VSCode_Projects/lidar_new/src/hikrobot_camera/script/location.py --color {color_value}'])
            # messagebox.showinfo("Success", "标定已启动")
        except Exception as e:
            messagebox.showerror("Error", f"标定时出错: {e}")

    def launch(self):
        color_value = "0" if self.color.get() == "red" else "1"
        try:
            subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', f'/home/qianzezhong/miniconda3/envs/yolov8/bin/python /home/qianzezhong/Documents/VSCode_Projects/lidar_new/src/hikrobot_camera/script/main.py \
--engine /home/qianzezhong/Documents/VSCode_Projects/lidar_new/best3.engine \
--device cuda:0 \
--color {color_value} \
--record 0 \
--armor_engine /home/qianzezhong/Documents/VSCode_Projects/lidar_new/armor_best.engine'])
            # messagebox.showinfo("Success", "启动已开始")
        except Exception as e:
            messagebox.showerror("Error", f"启动时出错: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
"""
/home/qianzezhong/miniconda3/envs/yolov8/bin/python /home/qianzezhong/Documents/VSCode_Projects/lidar_new/src/hikrobot_camera/script/main.py \
--engine /home/qianzezhong/Documents/VSCode_Projects/lidar_new/best3.engine \
--device cuda:0 \
--color 0 \
--record 0 \
--armor_engine /home/qianzezhong/Documents/VSCode_Projects/lidar_new/armor_best.engine

"""