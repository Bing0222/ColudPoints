import cv2
import numpy as np 


data = open("n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542800855949460.pcd.bin", "rb").read()

# Data is stored as (x, y, z, intensity, ring index).
pc = np.frombuffer(data, dtype=np.float32).reshape(-1, 5)

size = 1024
halfsize = size / 2
image = np.zeros((size, size, 3), np.uint8)
points = pc[:, :4] * [halfsize / 100, halfsize / 100, 1/15, 1] + [halfsize, halfsize, 0.5, 1]

for x, y, z, intensity in points:
    x = int(x)
    y = int(y)
    if x >=0 and x < size and y >= 0 and y < size:
        alpha = min(max(0, z) * 0.7 + 0.3, 1)
        color = alpha * 255

        image[y, x] = color

cv2.imwrite("points.jpg", image)