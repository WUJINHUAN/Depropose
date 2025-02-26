import numpy as np
from torch.utils.data import Dataset
import json
from PIL import Image
import cv2
from utils.transform import fliplr_joints, affine_transform, get_affine_transform
import os.path as osp
from torchvision import transforms
from tqdm import tqdm
from utils.position_encoding import get_rays_new
import time
import random
from noise.bandian import add_speckle_noise
from noise.gaosi import add_gaussian_noise
from noise.jiaoyan import add_salt_and_pepper_noise
from noise.quanhei import add_qunahei_noise
import os


class HumanPoseEstimationDataset(Dataset):
    """
    HumanPoseEstimationDataset class.

    Generic class for HPE datasets.
    """

    def __init__(self, annotation_path, dataset_root):
        """
        构造函数

        参数:
            data (list): 数据集的样本列表
            transform (callable, optional): 应用于样本的数据转换函数，默认为None.
        """
        self.annotation_path = annotation_path
        self.image_size = (224, 224)
        self.num_joints = 17
        self.dataset_root = dataset_root
        self.pixel_std = 200

        self.image_size = (224, 224)
        self.aspect_ratio = 1.0
        self.nums = 0

        self.noise = False

        self.H, self.W = 224, 224
        self.nums = 0
        self.noise = False
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        with open(self.annotation_path, 'r') as f:
            dataset = json.load(f)

        self.data = []
        for action in tqdm(dataset["images"]):
            for ca_01, ca_02, ca_03, ca_04 in zip(dataset["images"][action]["ca_01"],
                                                  dataset["images"][action]["ca_02"],
                                                  dataset["images"][action]["ca_03"],
                                                  dataset["images"][action]["ca_04"]):
                self.data.append(ca_01)
                self.data.append(ca_02)
                self.data.append(ca_03)
                self.data.append(ca_04)

        # for action in dataset["images"]:
        #     for camera in dataset["images"][action]:
        #         for data in dataset["images"][action][camera]:
        #             self.data.append(data)
        # for file1, file2 in zip(dataset["images"][action]["ca_01"], dataset["images"][action]["ca_03"]):
        #     # print(file1["image"])
        #     # print(file2["image"])
        #     self.data.append(file1)
        #     self.data.append(file2)
        #
        # for file1, file2 in zip(dataset["images"][action]["ca_02"], dataset["images"][action]["ca_04"]):
        #     # print(file1["image"])
        #     # print(file2["image"])
        #     self.data.append(file1)
        #     self.data.append(file2)

        # self.data.append({
        #     'imgId': imgId,
        #     'annId': obj['id'],
        #     'imgPath': f"{self.root_path}/{self.data_version}/{file_name}.jpg",
        #     # 'imgPath': os.path.join(self.root_path, self.data_version, 'id_%s.jpg' % imgId),
        #     'center': center,
        #     'scale': scale,
        #     'joints': joints,
        #     'joints_visibility': joints_visibility,
        #     'bbox': obj['clean_bbox'],
        #     'img_size': (img['width'], img['height']),
        #     'file_name': f"{file_name}.jpg"
        # })
        print("图片数量为:", len(self.data))
        print('\nHuman3.6 dataset loaded!')

    def __len__(self):
        return len(self.data)
        # 返回数据集的样本数量

    def __getitem__(self, index):
        """
        返回给定索引的样本

        参数:
            idx (int): 样本的索引

        返回:
            sample (dict): 样本的字典，包含图像数据和对应的姿态标注
        """
        joints_data = self.data[index].copy()

        path = joints_data['image']
        parts = path.split("\\")
        imgPath = parts[-1]
        img_path = osp.join(self.dataset_root, imgPath)

        try:
            #image = np.array(self.add_noise(img_path))
            image = np.array(Image.open(img_path))
        except:
            raise ValueError(f"Fail to read {joints_data['image']}")

        joints_3d = np.array(joints_data['joints_3d'])

        camera = joints_data['camera']

        try:
            K = np.array(camera['K'])
        except KeyError:
            print(f"Missing key 'K' in camera data: {img_path}")
            raise

        R = np.array(camera['R'])
        T = np.array(camera['T'])
        rays_d = get_rays_new(self.image_size, self.H, self.W, K, R, T).squeeze()

        x1, y1, x2, y2 = joints_data['box']
        box = [x1, y1, x2 - x1, y2 - y1]
        # box = np.array(box)
        # print(box)

        center, scale = self._box2cs(box)

        c = center
        s = scale

        score = joints_data['score'] if 'score' in joints_data else 1
        r = 0

        trans = get_affine_transform(c, s, self.pixel_std, r, self.image_size)
        image_transformed = cv2.warpAffine(
            image,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR
        )

        joints = [row[:2] for row in joints_3d]

        for i in range(self.num_joints):
            joints[i] = affine_transform(joints[i], trans)

        for i in range(len(joints_3d)):
            joints_3d[i][:2] = joints[i]

        image_transformed = self.transform(image_transformed)

        return image_transformed, joints_3d.astype(np.float32), rays_d, img_path

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2,), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def add_noise(self, image_path):
        image = Image.open(image_path)

        if (self.nums < 4 and self.noise == False):
            random_number = random.random()

            if (random_number < 0.25):
                self.noise = True
                noise_type = random.choice(['salt_and_pepper', 'gaussian', 'speckle'])

                if (noise_type == 'salt_and_pepper'):
                    image = add_salt_and_pepper_noise(image)
                if (noise_type == 'gaussian'):
                    image = add_gaussian_noise(image)
                if (noise_type == 'speckle'):
                    image = add_speckle_noise(image)
                # noisy_image.save(f'noisy_{noise_type}_{os.path.basename(image_path)}')
        self.nums += 1

        if (self.nums == 4):
            self.nums = 0
            self.noise = False
        return image

    def normalization_stats(self, complete_data, dim=3, predict_14=False):

        data_mean = np.mean(complete_data, axis=0)
        data_std = np.std(complete_data, axis=0)

        # Encodes which 17 (or 14) 2d-3d pairs we are predicting

        return data_mean, data_std

    def normalize_data(self, data, data_mean, data_std):
        """Normalizes a dictionary of poses

        Args
          data: dictionary where values are
          data_mean: np vector with the mean of the data
          data_std: np vector with the standard deviation of the data
          dim_to_use: list of dimensions to keep in the data
        Returns
          data_out: dictionary with same keys as data, but values have been normalized
        """
        if data.shape[1] != len(data_mean) or data.shape[1] != len(data_std):
            raise ValueError("Data dimensions do not match the mean and std vectors.")

        # 标准化数据
        data_out = (data - data_mean) / data_std
        return data_out
