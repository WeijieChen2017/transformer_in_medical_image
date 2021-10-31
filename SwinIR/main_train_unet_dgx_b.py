import argparse
import cv2
import glob
import time
import random
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from sys import getsizeof
import os
import torch
import requests

from models.network_swinir import SwinIR as net
from utils import util_calculate_psnr_ssim as util
from unet import UNet

np.random.seed(seed=813)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='color_dn', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    
    parser.add_argument('--tag', type=str, default="MR2CT_B_", help='Save_prefix')
    parser.add_argument('--gpu_ids', type=str, default="7", help='Use which GPU to train')
    parser.add_argument('--epoch', type=int, default=100, help='how many epochs to train')
    parser.add_argument('--batch', type=int, default=1, help='how many batches in one run')
    parser.add_argument('--loss_display_per_iter', type=int, default=600, help='display how many losses per iteration')
    parser.add_argument('--folder_pet', type=str, default="./MR2CT_B/X/train/", help='input folder of T1MAP images')
    parser.add_argument('--folder_sct', type=str, default="./MR2CT_B/Y/train/", help='input folder of BRAVO images')
    parser.add_argument('--folder_pet_v', type=str, default="./MR2CT_B/X/val/", help='input folder of T1MAP PET images')
    parser.add_argument('--folder_sct_v', type=str, default="./MR2CT_B/Y/val/", help='input folder of BRAVO images')
    
    args = parser.parse_args()

    gpu_list = ','.join(str(x) for x in args.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(args.model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {args.model_path}')
        open(args.model_path, 'wb').write(r.content)

    model = UNet(n_channels=3, n_classes=2, bilinear=True)
    model.train().float()
    model = model.to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    sct_list = sorted(glob.glob(args.folder_sct+"*.npy"))
    sct_list_v = sorted(glob.glob(args.folder_sct_v+"*.npy"))
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
            cube_x_data = np.load(cube_x_path)
            cube_y_data = np.load(cube_y_path)
            assert cube_x_data.shape == cube_y_data.shape
            len_z = cube_x_data.shape[1]
            case_loss = np.zeros((len_z//args.batch))
            input_list = list(range(len_z))
            random.shuffle(input_list)

            # 0:[32, 45, 23, 55], 1[76, 74, 54, 99], 3[65, 92, 28, 77], ...
            for idx_iter in range(len_z//args.batch):

                batch_x = np.zeros((args.batch, 3, cube_x_data.shape[0], cube_x_data.shape[2]))
                batch_y = np.zeros((args.batch, 3, cube_y_data.shape[0], cube_y_data.shape[2]))

                for idx_batch in range(args.batch):
                    z_center = input_list[idx_iter*args.batch+idx_batch]
                    batch_x[idx_batch, 1, :, :] = cube_x_data[:, z_center, :]
                    batch_y[idx_batch, 1, :, :] = cube_y_data[:, z_center, :]
                    z_before = z_center - 1 if z_center > 0 else 0
                    z_after = z_center + 1 if z_center < len_z-1 else len_z-1
                    batch_x[idx_batch, 0, :, :] = cube_x_data[:, z_before, :]
                    batch_y[idx_batch, 0, :, :] = cube_y_data[:, z_before, :]
                    batch_x[idx_batch, 2, :, :] = cube_x_data[:, z_after, :]
                    batch_y[idx_batch, 2, :, :] = cube_y_data[:, z_after, :]

                batch_x = torch.from_numpy(batch_x).float().to(device)
                batch_y = torch.from_numpy(batch_y).float().to(device)
                # print(batch_x.shape, batch_y.shape)
                # print(getsizeof(batch_x), getsizeof(batch_y))

                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
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
            np.save(args.tag+"Epoch[{:03d}]_Case[{}]_t.npy".format(idx_epoch+1, case_name),
                    (batch_x.cpu().detach().numpy(),
                     batch_y.cpu().detach().numpy(),
                     model(batch_x).cpu().detach().numpy()))

            # after training one case
            loss_mean = np.mean(case_loss)
            loss_std = np.std(case_loss)
            print("===>===> Epoch[{:03d}]-Case[{:03d}]: ".format(idx_epoch+1, cnt_sct+1), end='')
            print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))
            epoch_loss[cnt_sct] = loss_mean

        # after training all cases
        loss_mean = np.mean(epoch_loss)
        loss_std = np.std(epoch_loss)
        print("===>===>===> Epoch[{}]: ".format(idx_epoch+1), end='')
        print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))
        np.save(args.tag+"epoch_loss_{:03d}.npy".format(idx_epoch+1), epoch_loss)
        train_loss[idx_epoch] = loss_mean
        # ====================================>train<====================================

        # ====================================>val<====================================
        model.eval()
        random.shuffle(sct_list_v)
        for cnt_sct, sct_path in enumerate(sct_list_v):

            # train
            cube_x_path = sct_path.replace("Y", "X")
            cube_y_path = sct_path
            print("--->",cube_x_path,"<---", end="")
            cube_x_data = np.load(cube_x_path)
            cube_y_data = np.load(cube_y_path)
            assert cube_x_data.shape == cube_y_data.shape
            len_z = cube_x_data.shape[1]
            case_loss = np.zeros((len_z//args.batch))
            input_list = list(range(len_z))
            random.shuffle(input_list)

            # 0:[32, 45, 23, 55], 1[76, 74, 54, 99], 3[65, 92, 28, 77], ...
            for idx_iter in range(len_z//args.batch):

                batch_x = np.zeros((args.batch, 3, cube_x_data.shape[0], cube_x_data.shape[2]))
                batch_y = np.zeros((args.batch, 3, cube_y_data.shape[0], cube_y_data.shape[2]))

                for idx_batch in range(args.batch):
                    z_center = input_list[idx_iter*args.batch+idx_batch]
                    batch_x[idx_batch, 1, :, :] = cube_x_data[:, z_center, :]
                    batch_y[idx_batch, 1, :, :] = cube_y_data[:, z_center, :]
                    z_before = z_center - 1 if z_center > 0 else 0
                    z_after = z_center + 1 if z_center < len_z-1 else len_z-1
                    batch_x[idx_batch, 0, :, :] = cube_x_data[:, z_before, :]
                    batch_y[idx_batch, 0, :, :] = cube_y_data[:, z_before, :]
                    batch_x[idx_batch, 2, :, :] = cube_x_data[:, z_after, :]
                    batch_y[idx_batch, 2, :, :] = cube_y_data[:, z_after, :]

                batch_x = torch.from_numpy(batch_x).float().to(device)
                batch_y = torch.from_numpy(batch_y).float().to(device)
                
                loss = criterion(model(batch_x), batch_y)
                case_loss[idx_iter] = loss.item()
            
            # save one progress shot
            case_name = os.path.basename(cube_x_path)[4:7]
            np.save(args.tag+"Epoch[{:03d}]_Case[{}]_v.npy".format(idx_epoch+1, case_name),
                    (batch_x.cpu().detach().numpy(),
                     batch_y.cpu().detach().numpy(),
                     model(batch_x).cpu().detach().numpy()))
            
            # after training one case
            loss_mean = np.mean(case_loss)
            loss_std = np.std(case_loss)
            print("===>===> Epoch[{:03d}]-Val-Case[{:03d}]: ".format(idx_epoch+1, cnt_sct+1), end='')
            print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))
            epoch_loss_v[cnt_sct] = loss_mean

        loss_mean = np.mean(epoch_loss_v)
        loss_std = np.std(epoch_loss_v)
        print("===>===>===> Epoch[{:03d}]-Val: ".format(idx_epoch+1), end='')
        print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))
        np.save(args.tag+"epoch_loss_v_{:03d}.npy".format(idx_epoch+1), epoch_loss_v)
        if loss_mean < best_val_loss:
            # save the best model
            torch.save(model, args.tag+"model_best_{:03d}.pth".format(idx_epoch+1))
            print("Checkpoint saved at Epoch {:03d}".format(idx_epoch+1))
            best_val_loss = loss_mean
        # ====================================>val<====================================

    loss_mean = np.mean(train_loss)
    loss_std = np.std(train_loss)
    print("===>===>===>===>Training finished: ", end='')
    print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))

if __name__ == '__main__':
    main()
