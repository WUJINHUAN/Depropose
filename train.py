from torch.utils.data import DataLoader

from dataset.new_dataset import HumanPoseEstimationDataset
from tqdm import tqdm
import time
from model.pose3D_model import Pose3D
from utils.loss import mpjpe
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

from utils.normalization import unNormalizeData
from torch.cuda.amp import autocast, GradScaler
from utils.climb_lengths import calculate_length_error

def main():
    torch.backends.cudnn.benchmark = True
    cfg = config
    numpy.set_printoptions(suppress=True)
    pandas.set_option('display.float_format', lambda x: '%.2f' % x)

    work_dir = r'D:\Pythoncoda\new_model\output'

    log_file, session_dir = get_logfile(work_dir)
    logger = get_root_logger(log_file=log_file)

    dataset = HumanPoseEstimationDataset(annotation_path=r"D:\Pythoncoda\new_model\dataset\action1_all.json",
                                         dataset_root=r"D:\Pythoncoda\new_model\dataset\train2017")
    dataloaders_train = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False, pin_memory=False)

    model = Pose3D()
    model = model.to('cuda')

    scaler = GradScaler()
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_scheduler(cfg, optimizer, len(dataloaders_train))

    start_epoch = 0

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'''\n
        #========= [Train Configs] =========#
        # - BASE_LR: {cfg.TRAIN['BASE_LR']: .10f}
        # - WARMUP_EPOCHS: {cfg.TRAIN['WARMUP_EPOCHS']: .6f}
        # - DECAY_EPOCHS: {cfg.TRAIN['LR_SCHEDULER']['DECAY_EPOCHS']: .6f}
        # - MIN_LR: {cfg.TRAIN['MIN_LR']: .15f}
        # - WARMUP_LR: {cfg.TRAIN['WARMUP_LR']: .15f}
        # - Num params: {total_params:,d}
        #===================================# 
        ''')

    print("开始训练")
    losses_3d_train = []
    imag = torch.randn(1, 3, 224, 224)
    imag1 = torch.randn(1, 17, 3)
    imag2 = torch.randn(1, 224, 224, 3)
    for epoch in range(start_epoch, cfg.TRAIN['EPOCHS']):
        train_tqdm = tqdm(dataloaders_train)
        start_time = time.time()
        epoch_loss_3d_train = 0
        epoch_loss_3d_train_spatial = 0
        loss_total = 0
        N = 0
        accuracy = 0
        accuracy2 = 0

        for batch_idx, batch in enumerate(train_tqdm):
            optimizer.zero_grad()
            loss = 0
            images, targets, rays_d, img_path = batch

            images, targets, rays_d = images.to('cuda'), targets.to('cuda'), rays_d.to('cuda')

            targets_mean = torch.mean(targets, dim=0).to('cuda')
            targets_std = torch.std(targets, dim=0).to('cuda')
            targets = normalize_data(targets, targets_mean, targets_std)

            if (images.shape[0] != 32):
                continue

            with autocast():

                coords, loss = model(images, rays_d, targets, loss)

                init_new_targets = unNormalizeData(targets, targets_mean, targets_std)
                init_output = unNormalizeData(coords, targets_mean, targets_std)

                epoch_loss_3d_train += loss
                loss = loss + mpjpe(init_output, init_output)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            accuracy = accuracy + mpjpe(init_new_targets, init_output)

            lr_scheduler.step_update(epoch * len(dataloaders_train) + batch_idx)

            loss_total = loss_total + loss.item()
            N += 1

            train_tqdm.update()

        train_tqdm.refresh()
        train_tqdm.close()

        elapsed = (time.time() - start_time)
        logger.info('%d/%d time %.2f LR %.15f 3d_train %f  %f accuracy %f  accuracy_s2 %f' % (
            epoch + 1,
            cfg.TRAIN['EPOCHS'],
            elapsed / 60,
            optimizer.param_groups[0]['lr'],
            loss_total / N,
            epoch_loss_3d_train / N,
            # epoch_loss_3d_train_spatial / N,
            accuracy / N,
            accuracy2 / N
        ))

        ckpt_name = f"epoch{str(epoch).zfill(3)}.pth"
        ckpt_path = osp.join(session_dir, ckpt_name)
        if (epoch % 5 == 0):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)

    print("训练结束")


if __name__ == '__main__':
    main()
