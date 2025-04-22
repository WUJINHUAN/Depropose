import os
import random
from PIL import Image
import numpy as np
import cv2
import random
from PIL import Image, ImageDraw

parent_folder = r'E:\noise\pythonProject\missing2'  # 修改为你的父文件夹路径
folders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]

# 将文件夹按顺序分成三组，每组四个文件夹
groups = [folders[i:i + 4] for i in range(0, len(folders), 4)]

for group in groups:
    selected_folder = random.choice(group)  # 随机选择组中的一个文件夹
    print(selected_folder)
    for img_file in os.listdir(selected_folder):
        img_path = os.path.join(selected_folder, img_file)
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(img_path)
            # 获取图像的宽度和高度
            width, height = img.size

            # 创建一个 ImageDraw 对象
            draw = ImageDraw.Draw(img)
            # 定义黑色方块的尺寸
            square_size = 200
            # 设置要添加方块的数量
            num_squares = 10
            # 随机添加多个小黑方块
            for _ in range(num_squares):
                # 随机生成方块的左上角位置
                x_position = random.randint(0, width - square_size)
                y_position = random.randint(0, height - square_size)

                # 在随机位置绘制小黑方块
                draw.rectangle([x_position, y_position, x_position + square_size, y_position + square_size],
                               fill="black")
            # 保存修改后的图像，覆盖原图
                img.save(img_path)