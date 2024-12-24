
import cv2
import numpy as np

# 初始化变量
points = []
polygons = []
current_polygon = []
heights = []
map_width, map_height = 28.0, 15.0  # 实际地图大小

# 读取并调整地图图像大小
image = cv2.imread('/home/qianzezhong/Documents/VSCode_Projects/lidar_new/src/hikrobot_camera/script/static/map.png')
resized_image = cv2.resize(image, (560, 300))

def mouse_callback(event, x, y, flags, param):
    global points, current_polygon, resized_image
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        real_x = x / 560 * map_width
        real_y = y / 300 * map_height
        current_polygon.append((real_x, real_y))
        cv2.circle(resized_image, (x, y), 3, (0, 255, 0), -1)
        if len(current_polygon) > 1:
            x1, y1 = points[-2]
            x2, y2 = points[-1]
            cv2.line(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imshow('Map', resized_image)

cv2.namedWindow('Map')
cv2.setMouseCallback('Map', mouse_callback)

while True:
    cv2.imshow('Map', resized_image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        if points:
            print(f"Point confirmed: {points[-1]}")
    elif key == ord('z'):
        if points:
            removed_point = points.pop()
            current_polygon.pop()
            resized_image = cv2.resize(image, (560, 300))
            for polygon in polygons:
                for i in range(len(polygon)):
                    x1, y1 = int(polygon[i][0] / map_width * 560), int(polygon[i][1] / map_height * 300)
                    x2, y2 = int(polygon[(i + 1) % len(polygon)][0] / map_width * 560), int(polygon[(i + 1) % len(polygon)][1] / map_height * 300)
                    cv2.line(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            for point in points:
                cv2.circle(resized_image, point, 3, (0, 255, 0), -1)
                for i in range(len(points) - 1):
                    cv2.line(resized_image, points[i], points[i + 1], (0, 255, 0), 1)
            cv2.imshow('Map', resized_image)
    elif key == ord('p'):
        if current_polygon:
            x1, y1 = int(current_polygon[-1][0] / map_width * 560), int(current_polygon[-1][1] / map_height * 300)
            x2, y2 = int(current_polygon[0][0] / map_width * 560), int(current_polygon[0][1] / map_height * 300)
            cv2.line(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            polygons.append(current_polygon.copy())
            current_polygon = []
            cv2.imshow('Map', resized_image)
            height = input("Enter the height for the current polygon: ")
            heights.append(float(height))
    elif key == ord('e'):
        break

# 保存结果到文件
with open('polygons.txt', 'w') as f:
    for polygon, height in zip(polygons, heights):
        f.write(f"Polygon: {polygon}, Height: {height}\n")

cv2.destroyAllWindows()
