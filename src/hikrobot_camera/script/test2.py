import cv2
from test_location import build_lookup_table
import numpy as np
def lookup_real_world_coordinates(lookup_table, pixel_coord):
    x, y = pixel_coord
    coord = lookup_table[y//10, x//10]
    if coord[0] == 999:
        return (999, 999)
    else:
        return (coord[0], coord[1])

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        real_world_coord = lookup_real_world_coordinates(param, (x*2, y*2))
        print(f"Pixel coordinate: ({x*2}, {y*2}), Real world coordinate: {real_world_coord}")
# build_lookup_table(0,0)
# build_lookup_table(1,0)
Mode=1
if Mode==0:
    lookup_table=np.load('lookup_table_main.npy')
else:
    lookup_table=np.load('lookup_table_sub.npy')
if Mode==0:
    image = cv2.imread("/home/qianzezhong/Pictures/main.bmp")
else:
    image = cv2.imread("/home/qianzezhong/Pictures/sub.bmp")
resized_image = cv2.resize(image, (3072, 2048))
for i in range(lookup_table.shape[0]):
    for j in range(lookup_table.shape[1]):
        if lookup_table[i][j][0]<100:
            cv2.circle(resized_image,(j*10,i*10),1,(0,255-lookup_table[i][j][0]*10,lookup_table[i][j][0]*10),-1)

# 设置鼠标回调
cv2.namedWindow('Projected Image')
cv2.setMouseCallback('Projected Image', mouse_callback, lookup_table)

# 显示图像
cv2.imshow('Projected Image', cv2.resize(resized_image, (resized_image.shape[1] // 2, resized_image.shape[0] // 2)))
cv2.waitKey(0)
cv2.destroyAllWindows()