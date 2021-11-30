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
from unet import UNet_bridge

np.random.seed(seed=813)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_channel', type=int, default=3, help='the number of input channel')
    parser.add_argument('--output_channel', type=int, default=1, help='the number of output channel')
    parser.add_argument('--tag', type=str, default="./bridge_3000/MR-tf6-CT/", help='Save_prefix')
    parser.add_argument('--gpu_ids', type=str, default="6", help='Use which GPU to train')
    parser.add_argument('--epoch', type=int, default=50, help='how many epochs to train')
    parser.add_argument('--batch', type=int, default=10, help='how many batches in one run')
    parser.add_argument('--loss_display_per_iter', type=int, default=600, help='display how many losses per iteration')
    parser.add_argument('--folder_pet', type=str, default="./bridge_3000/X/train/", help='input folder of T1MAP images')
    parser.add_argument('--folder_sct', type=str, default="./bridge_3000/Y/train/", help='input folder of BRAVO images')
    parser.add_argument('--folder_pet_v', type=str, default="./bridge_3000/X/val/", help='input folder of T1MAP PET images')
    parser.add_argument('--folder_sct_v', type=str, default="./bridge_3000/Y/val/", help='input folder of BRAVO images')
    args = parser.parse_args()
    input_channel = args.input_channel
    output_channel = args.output_channel

    gpu_list = ','.join(str(x) for x in args.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for path in [args.tag, args.tag+"npy/", args.tag+"loss/"]:
        if not os.path.exists(path):
            os.mkdir(path)

    model = UNet_bridge(n_channels=3, n_classes=1, bilinear=True, pre_train=True)

    model_MR = torch.load("./bridge_3000/MR/model_best_028.pth")
    model_CT = torch.load("./bridge_3000/CT/model_best_050.pth")
    modules_MR = ["inc", "down1", "down2", "down3", "down4", "hidden_1"]
    modules_CT = ["hidden_2", "up1", "up2", "up3", "up4", "outc"]
    model_dict_MR = model_MR.state_dict()
    model_dict_CT = model_CT.state_dict()
    new_state_dict_MR = {}
    new_state_dict_CT = {}

    for keys_MR in modules_MR:
        for k, v in model_dict_MR.items():
            if keys_MR in k:
                new_state_dict_MR[k] = v

    for keys_CT in modules_CT:
        for k, v in model_dict_CT.items():
            if keys_CT in k:
                new_state_dict_CT[k] = v

    model_state_dict_MR = model.state_dict()
    model_state_dict_CT = model.state_dict()
    model_state_dict_MR.update(new_state_dict_MR)
    model_state_dict_CT.update(new_state_dict_CT)
    model.load_state_dict(model_state_dict_MR)
    model.load_state_dict(model_state_dict_CT)

    model_state_dict = model.state_dict()
    dict_name = list(model_state_dict)
    for i, p in enumerate(dict_name):
        print(i, p)

    model.train().float()
    model = model.to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

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
            # cube_y_path = sct_path.replace("Y", "X")
            # cube_x_path = sct_path
            cube_y_path = sct_path
            print("--->",cube_x_path,"<---", end="")
            # cube_x_data = np.load(cube_x_path)
            # cube_y_data = np.load(cube_y_path)
            cube_x_data = nib.load(cube_x_path).get_fdata()
            cube_y_data = nib.load(cube_y_path).get_fdata()
            len_z = cube_x_data.shape[2]
            case_loss = np.zeros((len_z//args.batch))
            input_list = list(range(len_z))
            random.shuffle(input_list)

            # 0:[32, 45, 23, 55], 1[76, 74, 54, 99], 3[65, 92, 28, 77], ...
            for idx_iter in range(len_z//args.batch):

                batch_x = np.zeros((args.batch, input_channel, cube_x_data.shape[0], cube_x_data.shape[1]))
                batch_y = np.zeros((args.batch, output_channel, cube_y_data.shape[0], cube_y_data.shape[1]))

                for idx_batch in range(args.batch):
                    z_center = input_list[idx_iter*args.batch+idx_batch]
                    z_before = z_center - 1 if z_center > 0 else 0
                    z_after = z_center + 1 if z_center < len_z-1 else len_z-1
                    batch_x[idx_batch, 1, :, :] = cube_x_data[:, :, z_center]
                    batch_x[idx_batch, 0, :, :] = cube_x_data[:, :, z_before]
                    batch_x[idx_batch, 2, :, :] = cube_x_data[:, :, z_after]
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
            # np.save(args.tag+"npy/Epoch[{:03d}]_Case[{}]_t_x.npy".format(idx_epoch+1, case_name), batch_x.cpu().detach().numpy())
            # np.save(args.tag+"npy/Epoch[{:03d}]_Case[{}]_t_y.npy".format(idx_epoch+1, case_name), batch_y.cpu().detach().numpy())
            # np.save(args.tag+"npy/Epoch[{:03d}]_Case[{}]_t_z.npy".format(idx_epoch+1, case_name), y_hat.cpu().detach().numpy())

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
        np.save(args.tag+"loss/epoch_loss_t_{:03d}.npy".format(idx_epoch+1), epoch_loss)
        train_loss[idx_epoch] = loss_mean
        torch.cuda.empty_cache()
        # ====================================>train<====================================

        # ====================================>val<====================================
        model.eval()
        random.shuffle(sct_list_v)
        for cnt_sct, sct_path in enumerate(sct_list_v):

            # eval
            cube_x_path = sct_path.replace("Y", "X")
            # cube_y_path = sct_path.replace("Y", "X")
            # cube_x_path = sct_path
            cube_y_path = sct_path
            print("--->",cube_x_path,"<---", end="")
            # cube_x_data = np.load(cube_x_path)
            # cube_y_data = np.load(cube_y_path)
            cube_x_data = nib.load(cube_x_path).get_fdata()
            cube_y_data = nib.load(cube_y_path).get_fdata()
            len_z = cube_x_data.shape[2]
            case_loss = np.zeros((len_z//args.batch))
            input_list = list(range(len_z))
            random.shuffle(input_list)

            # 0:[32, 45, 23, 55], 1[76, 74, 54, 99], 3[65, 92, 28, 77], ...
            for idx_iter in range(len_z//args.batch):

                batch_x = np.zeros((args.batch, input_channel, cube_x_data.shape[0], cube_x_data.shape[1]))
                batch_y = np.zeros((args.batch, output_channel, cube_y_data.shape[0], cube_y_data.shape[1]))

                for idx_batch in range(args.batch):
                    z_center = input_list[idx_iter*args.batch+idx_batch]
                    z_before = z_center - 1 if z_center > 0 else 0
                    z_after = z_center + 1 if z_center < len_z-1 else len_z-1
                    batch_x[idx_batch, 1, :, :] = cube_x_data[:, :, z_center]
                    batch_x[idx_batch, 0, :, :] = cube_x_data[:, :, z_before]
                    batch_x[idx_batch, 2, :, :] = cube_x_data[:, :, z_after]
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
            np.save(args.tag+"npy/Epoch[{:03d}]_Case[{}]_v_x.npy".format(idx_epoch+1, case_name), batch_x.cpu().detach().numpy())
            np.save(args.tag+"npy/Epoch[{:03d}]_Case[{}]_v_y.npy".format(idx_epoch+1, case_name), batch_y.cpu().detach().numpy())
            np.save(args.tag+"npy/Epoch[{:03d}]_Case[{}]_v_z.npy".format(idx_epoch+1, case_name), y_hat.cpu().detach().numpy())

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
        np.save(args.tag+"loss/epoch_loss_v_{:03d}.npy".format(idx_epoch+1), epoch_loss_v)
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
