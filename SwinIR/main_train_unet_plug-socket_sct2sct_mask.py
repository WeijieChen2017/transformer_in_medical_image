import argparse
import cv2
import glob
import time
import random
import numpy as np
import nibabel as nib
import torch.nn as nn
from collections import OrderedDict
from sys import getsizeof
import os
import torch
import requests

# from models.network_swinir import SwinIR as net
from utils import util_calculate_psnr_ssim as util
from unet import UNet_simple

def generate_patch_seq(len_x, len_y, size_patch):
    mask_input = np.zeros((len_x, len_y))
    len_list_x = len_x // size_patch
    len_list_y = len_y // size_patch
    list_patch_x = list(np.arange(len_list_x))
    list_patch_y = list(np.arange(len_list_y))
    xy_mask = [None]*(len_list_x*len_list_y)
    cnt = 0
    for idx_x in range(len_list_x):
        for idx_y in range(len_list_y):
            xy_mask[cnt] = [idx_x, idx_y]
            cnt += 1
    return xy_mask

def generate_mask(len_x, len_y, xy_mask, mask_ratio, size_patch):
    random.shuffle(xy_mask)
    xy_mask_list = xy_mask[:int((1-mask_ratio)*len(xy_mask))]
    input_mask = np.zeros((len_x, len_y))
    for coord_couple in xy_mask_list:
        x, y = coord_couple
        x1 = x * size_patch
        x2 = x1 + size_patch
        y1 = y * size_patch
        y2 = y1 + size_patch
        input_mask[x1:x2, y1:y2] = 1
    
    return input_mask

def main():
    np.random.seed(seed=813)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_channel', type=int, default=3, help='the number of input channel')
    parser.add_argument('--output_channel', type=int, default=1, help='the number of output channel')
    parser.add_argument('--tag', type=str, default="./SQR/", help='Save_prefix')
    parser.add_argument('--gpu_ids', type=str, default="7", help='Use which GPU to train')
    parser.add_argument('--epoch', type=int, default=50, help='how many epochs to train')
    parser.add_argument('--batch', type=int, default=10, help='how many batches in one run')
    parser.add_argument('--loss_display_per_iter', type=int, default=600, help='display how many losses per iteration')
    parser.add_argument('--folder_pet', type=str, default="./SQR/X/train/", help='input folder of T1MAP images')
    parser.add_argument('--folder_sct', type=str, default="./SQR/Y/train/", help='input folder of BRAVO images')
    parser.add_argument('--folder_pet_v', type=str, default="./SQR/X/val/", help='input folder of T1MAP PET images')
    parser.add_argument('--folder_sct_v', type=str, default="./SQR/Y/val/", help='input folder of BRAVO images')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask_ratio of input')
    parser.add_argument('--mask_size_patch', type=int, default=16, help='size patch of masks')
    
    args = parser.parse_args()
    input_channel = args.input_channel
    output_channel = args.output_channel
    xy_mask = generate_patch_seq(len_x=512, len_y=512, size_patch=args.mask_size_patch)

    gpu_list = ','.join(str(x) for x in args.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet_simple(n_channels=3, n_classes=1, bilinear=True)
    model.train().float()
    model = model.to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    sct_list = sorted(glob.glob(args.folder_sct+"*.nii.gz"))
    sct_list_v = sorted(glob.glob(args.folder_sct_v+"*.nii.gz"))
    train_loss = np.zeros((args.epoch))
    epoch_loss = np.zeros((len(sct_list)))
    epoch_loss_v = np.zeros((len(sct_list_v)))
    best_val_loss = 1e6
    per_iter_loss = np.zeros((args.loss_display_per_iter))
    
    case_loss = None

    for idx_epoch in range(args.epoch):
        print("~~~~~~Epoch[{:03d}]~~~~~~".format(idx_epoch+1))

        # ====================================>train<====================================
        model.train()
        # randonmize training dataset
        random.shuffle(sct_list)
        for cnt_sct, sct_path in enumerate(sct_list):

            cube_x_path = sct_path.replace("Y", "X")
            cube_y_path = sct_path
            print("--->",cube_x_path,"<---", end="")
            # cube_x_data = np.load(cube_x_path)
            # cube_y_data = np.load(cube_y_path)
            cube_x_data = nib.load(cube_y_path).get_fdata()
            cube_y_data = nib.load(cube_y_path).get_fdata()
            len_z = cube_x_data.shape[2]
            case_loss = np.zeros((len_z//args.batch))
            input_list = list(range(len_z))
            random.shuffle(input_list)

            # 0:[32, 45, 23, 55], 1[76, 74, 54, 99], 3[65, 92, 28, 77], ...
            for idx_iter in range(len_z//args.batch):

                batch_x = np.zeros((args.batch, input_channel, cube_x_data.shape[0], cube_x_data.shape[1]))
                batch_y = np.zeros((args.batch, output_channel, cube_y_data.shape[0], cube_y_data.shape[1]))

                batch_x_mask = generate_mask(len_x=batch_x.shape[2],
                                             len_y=batch_x.shape[3],
                                             xy_mask=xy_mask,
                                             mask_ratio=args.mask_ratio,
                                             size_patch=args.mask_size_patch)

                for idx_batch in range(args.batch):
                    z_center = input_list[idx_iter*args.batch+idx_batch]
                    z_before = z_center - 1 if z_center > 0 else 0
                    z_after = z_center + 1 if z_center < len_z-1 else len_z-1
                    batch_x[idx_batch, 1, :, :] = np.multiply(cube_x_data[:, :, z_center], batch_x_mask)
                    batch_x[idx_batch, 0, :, :] = np.multiply(cube_x_data[:, :, z_before], batch_x_mask)
                    batch_x[idx_batch, 2, :, :] = np.multiply(cube_x_data[:, :, z_after], batch_x_mask)
                    if output_channel == 3:
                        batch_y[idx_batch, 0, :, :] = cube_y_data[:, :, z_before]
                        batch_y[idx_batch, 1, :, :] = cube_y_data[:, :, z_center]
                        batch_y[idx_batch, 2, :, :] = cube_y_data[:, :, z_after]
                    if output_channel == 1:
                        batch_y[idx_batch, 0, :, :] = cube_y_data[:, :, z_center]

                batch_x = torch.from_numpy(batch_x).float().to(device)
                batch_y = torch.from_numpy(batch_y).float().to(device)

                optimizer.zero_grad()
                y_hat = model(batch_x)
                loss = criterion(y_hat, batch_y)
                loss.backward()
                optimizer.step()

                per_iter_loss[idx_iter % args.loss_display_per_iter] = loss.item()
                case_loss[idx_iter] = loss.item()
                if idx_iter % args.loss_display_per_iter == args.loss_display_per_iter - 1:
                    loss_mean = np.mean(per_iter_loss)
                    loss_std = np.std(per_iter_loss)
                    print("===> Epoch[{:03d}]-Case[{:03d}]({:03d}/{:03d}): ".format(idx_epoch+1, cnt_sct+1, idx_iter + 1, len_z//args.batch), end='')
                    print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))

                case_loss[idx_iter] = loss.item()
            
            case_name = os.path.basename(cube_x_path)[4:7]
            # np.save(args.tag+"Epoch[{:03d}]_Case[{}]_t_x.npy".format(idx_epoch+1, case_name), batch_x.cpu().detach().numpy())
            # np.save(args.tag+"Epoch[{:03d}]_Case[{}]_t_y.npy".format(idx_epoch+1, case_name), batch_y.cpu().detach().numpy())
            # np.save(args.tag+"Epoch[{:03d}]_Case[{}]_t_z.npy".format(idx_epoch+1, case_name), y_hat.cpu().detach().numpy())

            # after training one case
            loss_mean = np.mean(case_loss)
            loss_std = np.std(case_loss)
            print("===> Epoch[{:03d}]-Case[{:03d}]: ".format(idx_epoch+1, cnt_sct+1), end='')
            print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))
            epoch_loss[cnt_sct] = loss_mean

        # after training all cases
        loss_mean = np.mean(epoch_loss)
        loss_std = np.std(epoch_loss)
        print("===> Epoch[{}]: ".format(idx_epoch+1), end='')
        print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))
        np.save(args.tag+"epoch_loss_t_{:03d}.npy".format(idx_epoch+1), epoch_loss)
        train_loss[idx_epoch] = loss_mean
        torch.cuda.empty_cache()
        # ====================================>train<====================================

        # ====================================>val<====================================
        model.eval()
        random.shuffle(sct_list_v)
        for cnt_sct, sct_path in enumerate(sct_list_v):

            # train
            cube_x_path = sct_path.replace("Y", "X")
            cube_y_path = sct_path
            print("--->",cube_x_path,"<---", end="")
            # cube_x_data = np.load(cube_x_path)
            # cube_y_data = np.load(cube_y_path)
            cube_x_data = nib.load(cube_y_path).get_fdata()
            cube_y_data = nib.load(cube_y_path).get_fdata()
            len_z = cube_x_data.shape[2]
            case_loss = np.zeros((len_z//args.batch))
            input_list = list(range(len_z))
            random.shuffle(input_list)

            # 0:[32, 45, 23, 55], 1[76, 74, 54, 99], 3[65, 92, 28, 77], ...
            for idx_iter in range(len_z//args.batch):

                batch_x = np.zeros((args.batch, input_channel, cube_x_data.shape[0], cube_x_data.shape[1]))
                batch_y = np.zeros((args.batch, output_channel, cube_y_data.shape[0], cube_y_data.shape[1]))

                batch_x_mask = generate_mask(len_x=batch_x.shape[2],
                                             len_y=batch_x.shape[3],
                                             xy_mask=xy_mask,
                                             mask_ratio=args.mask_ratio)

                for idx_batch in range(args.batch):
                    z_center = input_list[idx_iter*args.batch+idx_batch]
                    z_before = z_center - 1 if z_center > 0 else 0
                    z_after = z_center + 1 if z_center < len_z-1 else len_z-1
                    batch_x[idx_batch, 1, :, :] = np.multiply(cube_x_data[:, :, z_center], batch_x_mask)
                    batch_x[idx_batch, 0, :, :] = np.multiply(cube_x_data[:, :, z_before], batch_x_mask)
                    batch_x[idx_batch, 2, :, :] = np.multiply(cube_x_data[:, :, z_after], batch_x_mask)
                    if output_channel == 3:
                        batch_y[idx_batch, 0, :, :] = cube_y_data[:, :, z_before]
                        batch_y[idx_batch, 1, :, :] = cube_y_data[:, :, z_center]
                        batch_y[idx_batch, 2, :, :] = cube_y_data[:, :, z_after]
                    if output_channel == 1:
                        batch_y[idx_batch, 0, :, :] = cube_y_data[:, :, z_center]

                batch_x = torch.from_numpy(batch_x).float().to(device)
                batch_y = torch.from_numpy(batch_y).float().to(device)
                
                y_hat = model(batch_x)
                loss = criterion(y_hat, batch_y)
                case_loss[idx_iter] = loss.item()
            
            # save one progress shot
            case_name = os.path.basename(cube_x_path)[4:7]
            np.save(args.tag+"Epoch[{:03d}]_Case[{}]_v_x.npy".format(idx_epoch+1, case_name), batch_x.cpu().detach().numpy())
            np.save(args.tag+"Epoch[{:03d}]_Case[{}]_v_y.npy".format(idx_epoch+1, case_name), batch_y.cpu().detach().numpy())
            np.save(args.tag+"Epoch[{:03d}]_Case[{}]_v_z.npy".format(idx_epoch+1, case_name), y_hat.cpu().detach().numpy())

            # after training one case
            loss_mean = np.mean(case_loss)
            loss_std = np.std(case_loss)
            print("===> Epoch[{:03d}]-Val-Case[{:03d}]: ".format(idx_epoch+1, cnt_sct+1), end='')
            print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))
            epoch_loss_v[cnt_sct] = loss_mean

        loss_mean = np.mean(epoch_loss_v)
        loss_std = np.std(epoch_loss_v)
        print("===> Epoch[{:03d}]-Val: ".format(idx_epoch+1), end='')
        print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))
        np.save(args.tag+"epoch_loss_v_{:03d}.npy".format(idx_epoch+1), epoch_loss_v)
        if loss_mean < best_val_loss:
            # save the best model
            torch.save(model, args.tag+"model_best_{:03d}.pth".format(idx_epoch+1))
            print("Checkpoint saved at Epoch {:03d}".format(idx_epoch+1))
            best_val_loss = loss_mean
        torch.cuda.empty_cache()
        # ====================================>val<====================================

    loss_mean = np.mean(train_loss)
    loss_std = np.std(train_loss)
    print("===>Training finished: ", end='')
    print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))

if __name__ == '__main__':
    main()
