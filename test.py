from torch.utils.data import DataLoader, ConcatDataset

from dataset.new_dataset import HumanPoseEstimationDataset
from tqdm import tqdm
import time
from model.pose3D_model import Pose3D
from utils.loss import mpjpe, p_mpjpe
import torch.optim as optim
from utils.logging import get_root_logger
from utils.logging import get_logfile
import torch
import os.path as osp
import os
from glob import glob
import numpy
import pandas
from utils.optimizer import build_optimizer
from utils.lr_scheduler import build_scheduler
from config import config
from utils.normalization import normalize_data
from utils.normalization import normalization_stats
from utils.normalization import unNormalizeData
from torch.cuda.amp import autocast, GradScaler
from utils.transform import fliplr_joints, affine_transform, get_affine_transform
from utils.提取动作名称 import extract_action
import json
import numpy as np


def test():
    torch.set_printoptions(threshold=float('inf'))
    cfg = config
    numpy.set_printoptions(suppress=True)
    pandas.set_option('display.float_format', lambda x: '%.2f' % x)

    work_dir = r'D:\Pythoncoda\new_model\output_test'

    log_file, session_dir = get_logfile(work_dir)
    logger = get_root_logger(log_file=log_file)



    dataset_test2 = HumanPoseEstimationDataset(annotation_path=r"D:\Pythoncoda\new_model\dataset\action1_all.json",
                                         dataset_root=r"D:\Pythoncoda\new_model\dataset\train2017")


    dataloaders_test2 = DataLoader(dataset_test2, batch_size=32, num_workers=4, shuffle=False, pin_memory=False)
    test_tqdm2 = tqdm(dataloaders_test2)

    # 初始化模型并加载训练好的权重
    model = Pose3D()
    model = model.to('cuda')

    ckpt_path = r''  # 使用保存的训练好的模型权重
    checkpoint = torch.load(ckpt_path)
    pretrained_dict = checkpoint['state_dict']
    model.load_state_dict(pretrained_dict)

    model.eval()  # 设置为评估模式

    print("开始测试")

    total_mpjpe = 0  # 用于累计MPJPE


    N = 0  # 测试样本数
    loss = 0
    finall_loss = 0
    with torch.no_grad():  # 测试时不计算梯度

        for batch_idx, batch in enumerate(test_tqdm2):
            images, targets, rays_d, img_path = batch

            images, targets, rays_d, = images.to('cuda'), targets.to('cuda'), rays_d.to('cuda')

            targets_mean = torch.mean(targets, dim=0).to('cuda')
            targets_std = torch.std(targets, dim=0).to('cuda')
            norma_targets = normalize_data(targets, targets_mean, targets_std)

            if (images.shape[0] != 32):
                continue
            img_path_name = []
            for img in img_path:
                name = extract_action(img)
                img_path_name.append(name)
            if len(set(img_path_name)) != 1:
                continue

            with autocast():

                coords, loss = model(images, rays_d, norma_targets, loss)

            init_output = unNormalizeData(coords, targets_mean, targets_std)

            batch_mpjpe = mpjpe(init_output, targets)

            logger.info(f"四个视角平均误差:{batch_mpjpe}")
            logger.info("")

            total_mpjpe += batch_mpjpe

            N += 1

    avg_mpjpe = total_mpjpe / N
    # avg_p_mpjpe = total_p_mpjpe / N
    avg_mpjpe2 = finall_loss / N

    logger.info(f'Average MPJPE on the test set: {avg_mpjpe:.4f} mm 和 N={N}')
    # logger.info(f'Average P-MPJPE on the test set: {avg_p_mpjpe:.4f} mm 和 N={N}')
    logger.info(f'Average MPJPE on the test set: {avg_mpjpe2:.4f} mm 和 N={N}')

    numpy_array = targets
    import numpy as np
    np.set_printoptions(suppress=True, precision=6)
    # logger.info(f"Value: {numpy_array[0]}")
    # logger.info(f"Value: {numpy_array[1]}")

    print("测试结束")


if __name__ == '__main__':
    test()
