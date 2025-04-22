import os
import random
from PIL import Image
import numpy as np


# 定义噪声函数
def add_gaussian_noise(image, mean=0, std=25):
    """添加高斯噪声"""
    image_array = np.array(image)
    noise = np.random.normal(mean, std, image_array.shape)
    noisy_image_array = image_array + noise
    noisy_image_array = np.clip(noisy_image_array, 0, 255)
    return Image.fromarray(noisy_image_array.astype('uint8'))


def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """添加椒盐噪声"""
    image_array = np.array(image)
    output = np.copy(image_array)
    height, width, channels = image_array.shape

    # 盐噪声
    num_salt = int(salt_prob * height * width)
    salt_coords = [np.random.randint(0, i, num_salt) for i in image_array.shape[:2]]
    output[salt_coords[0], salt_coords[1], :] = 255

    # 椒噪声
    num_pepper = int(pepper_prob * height * width)
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in image_array.shape[:2]]
    output[pepper_coords[0], pepper_coords[1], :] = 0

    return Image.fromarray(output.astype('uint8'))


def add_speckle_noise(image, mean=0, std=25):
    """添加斑点噪声"""
    image_array = np.array(image)
    noise = np.random.normal(mean, std, image_array.shape)
    noisy_image_array = image_array + image_array * noise
    noisy_image_array = np.clip(noisy_image_array, 0, 255)
    return Image.fromarray(noisy_image_array.astype('uint8'))


# 噪声函数列表
noise_functions = [add_gaussian_noise, add_salt_and_pepper_noise, add_speckle_noise]

parent_folder = r'E:\noise\pythonProject\实验'  # 修改为你的父文件夹路径
folders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]

# 将文件夹按顺序分成三组，每组四个文件夹
groups = [folders[i:i + 4] for i in range(0, len(folders), 4)]

for group in groups:
    selected_folder = random.choice(group)  # 随机选择组中的一个文件夹
    print(selected_folder)
    for img_file in os.listdir(selected_folder):
        img_path = os.path.join(selected_folder, img_file)
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            noise_func = random.choice(noise_functions)
            image = Image.open(img_path)
            noisy_image = noise_func(image)  # 添加噪声
            noisy_image.save(img_path)  # 直接覆盖原图片

        #print(f"噪声已添加并覆盖原图片: {selected_folder}，噪声类型: {noise_func.__name__}")