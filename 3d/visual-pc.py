# point cloud

import numpy as np
import cv2
# import matplotlib.pyplot as plt

file = "n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542800855949460.pcd.bin"

#   x, y, z就是基于激光雷达的3d坐标系
#   intensity，反射强度（指的是，激光接收器接受的反射次数）
#        - 更多的是跟激光打上的物理性质有关
#   ring index，激光不是32线吗，他就是1-32
# [x, y, z, intensity, ring index]
pc = np.frombuffer(open(file, "rb").read(), dtype=np.float32)
pc = pc.reshape(-1, 5)[:, :4]
pc.tofile("kitti.format.pcd.bin")

x, y, z, intensity = pc.T

# 设置图像的尺寸1024x1024
image_size = 1024

# 把数据归一化
# 点的坐标范围大概是100
pc_range = 100
x = x / pc_range  # -1 到 +1
y = y / pc_range

# 缩放到图像大小，并平移到图像中心
half_image_size = image_size / 2
x = x * half_image_size + half_image_size
y = y * half_image_size + half_image_size

# opencv的图像，是可以用numpy进行创建的
image = np.zeros((image_size, image_size, 3), np.uint8)

for ix, iy, iz in zip(x, y, z):
    ix = int(ix)
    iy = int(iy)

    # 判断是否在图像范围内
    if ix >= 0 and ix < image_size and iy >= 0 and iy < image_size:

        alpha = min(1, max(0, (iz) / 5))
        color = alpha * 128 + 127
        image[iy, ix] = color, color, color

cv2.imwrite("my-pc.jpg", image)