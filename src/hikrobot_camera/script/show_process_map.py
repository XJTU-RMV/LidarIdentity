import cv2
import numpy as np

def read_polygons(file_path):
    polygons = []
    heights = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("Polygon:"):
                parts = line.strip().split(", Height: ")
                polygon_str = parts[0][9:].strip()
                height = float(parts[1].strip())
                polygon = eval(polygon_str)
                polygons.append(polygon)
                heights.append(height)
    return polygons, heights

def add_base_polygon(polygons, heights, width, height):
    base_polygon = [(0, 0), (width, 0), (width, height), (0, height)]
    polygons.insert(0, base_polygon)
    heights.insert(0, 0.0)

def adjust_polygon_y(polygons, offset):
    adjusted_polygons = []
    for polygon in polygons:
        adjusted_polygon = [(x, y - offset) for x, y in polygon]
        adjusted_polygons.append(adjusted_polygon)
    return adjusted_polygons

def draw_polygons(image, polygons):
    for polygon in polygons:
        points = np.array([[int(x / 28.0 * 560), int(y / 15.0 * 300)] for x, y in polygon], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=1)
    cv2.imshow('Polygons', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.float32), point, False) >= 0

def sample_height(polygons, heights, x_range, y_range, step):
    x_samples = np.arange(x_range[0], x_range[1]+step, step)
    y_samples = np.arange(y_range[0], y_range[1]+step, step)
    height_map = {}
    for x in x_samples:
        for y in y_samples:
            # y=-15-y
            # print(y)
            for polygon, height in sorted(zip(polygons, heights), key=lambda ph: ph[1], reverse=True):
                if is_point_in_polygon((x, y), polygon):
                    height_map[(x, -15-y)] = height
                    break
    return height_map

def save_height_map(file_path, height_map):
    with open(file_path, 'w') as f:
        for (x, y), height in height_map.items():
            f.write(f"({x:.2f}, {y:.2f}): {height}\n")
def process_map():
    map_width, map_height = 28.0, 15.0
    polygons, heights = read_polygons('polygons.txt')
    add_base_polygon(polygons, heights, map_width, map_height)
    adjusted_polygons = adjust_polygon_y(polygons, 15)

    image = cv2.imread('/home/qianzezhong/Documents/VSCode_Projects/lidar_new/src/hikrobot_camera/script/static/map.png')
    resized_image = cv2.resize(image, (560, 300))
    draw_polygons(resized_image, polygons)

    height_map = sample_height(adjusted_polygons, heights, (0, 28.1), (-15, 0.1), 0.1)
    # print(height_map.shape)
    save_height_map('height_map.txt', height_map)
if __name__=="__main__":
    process_map()