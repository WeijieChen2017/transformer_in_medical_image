import os
import cv2
import glob
import torch
import random
import argparse
import numpy as np
import nibabel as nib
import torch.nn as nn

from models.network_swinir import SwinIR as net
from transunet.vit_seg_modeling import VisionTransformer as ViT_seg
from transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from utils import util_calculate_psnr_ssim as util
from unet import UNet

np.random.seed(seed=813)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel')
    parser.add_argument('--output_channel', type=int, default=1, help='the number of output channel')
    parser.add_argument('--save_folder', type=str, default="./MR2CT_B_VIT/", help='Save_prefix')
    parser.add_argument('--gpu_ids', type=str, default="7", help='Use which GPU to train')
    parser.add_argument('--epoch', type=int, default=50, help='how many epochs to train')
    parser.add_argument('--batch', type=int, default=1, help='how many batches in one run')
    parser.add_argument('--loss_display_per_iter', type=int, default=600, help='display how many losses per iteration')
    parser.add_argument('--folder_train_x', type=str, default="./MR2CT_B_UNET/X/train/", help='input folder of trianing data X')
    parser.add_argument('--folder_train_y', type=str, default="./MR2CT_B_UNET/Y/train/", help='input folder of training data Y')
    parser.add_argument('--folder_val_x', type=str, default="./MR2CT_B_UNET/X/val/", help='input folder of validation data X')
    parser.add_argument('--folder_val_y', type=str, default="./MR2CT_B_UNET/Y/val/", help='input folder of validation data Y')
    parser.add_argument('--weights_path', type=str, default='./pretrain_models/deit_base_distilled_patch16_384.pth')
    
    args = parser.parse_args()
    input_channel = args.input_channel
    output_channel = args.output_channel

    gpu_list = ','.join(str(x) for x in args.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for path in [args.save_folder+"npy/"]:
        if not os.path.exists(path):
            os.mkdir(path)

    config_vit = CONFIGS_ViT_seg["R50-ViT-L_16"]
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    model = ViT_seg(config_vit, img_size=256, num_classes=1)
    
    model.train().float()
    model = model.to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    list_train_y = sorted(glob.glob(args.folder_train_y+"*.nii.gz"))
    list_val_y = sorted(glob.glob(args.folder_val_y+"*.nii.gz"))

    train_loss = np.zeros((args.epoch)) # over the whole process
    epoch_loss_t = np.zeros((len(list_train_y))) # over the training part of each epoch
    epoch_loss_v = np.zeros((len(list_val_y))) # over the validation part of each epoch
    best_val_loss = 1e6 # save the best model if the validation loss is less than this
    per_iter_loss = np.zeros((args.loss_display_per_iter)) # to show loss every X batch inside one case
    case_loss = None # the loss over one case

    for idx_epoch in range(args.epoch):
        print("~~~~~~Epoch[{:03d}]~~~~~~".format(idx_epoch+1))

        # ====================================>train<====================================
        model.train()
        # randonmize training dataset
        random.shuffle(list_train_y)
        for cnt_y, path_y in enumerate(list_train_y):

            case_x_path = path_y.replace("Y", "X")
            case_y_path = path_y
            print("->",case_x_path,"<-", end="")
            case_x_data = nib.load(case_x_path).get_fdata()
            case_y_data = nib.load(case_y_path).get_fdata()
            len_z = case_x_data.shape[2]
            case_loss = np.zeros((len_z//args.batch))
            input_list = list(range(len_z))
            random.shuffle(input_list)

            # 0:[32, 45, 23, 55], 1[76, 74, 54, 99], 3[65, 92, 28, 77], ...
            for idx_iter in range(len_z//args.batch):

                batch_x = np.zeros((args.batch, input_channel, case_x_data.shape[0], case_x_data.shape[1]))
                batch_y = np.zeros((args.batch, output_channel, case_y_data.shape[0], case_y_data.shape[1]))

                for idx_batch in range(args.batch):
                    z_center = input_list[idx_iter*args.batch+idx_batch]
                    z_before = z_center - 1 if z_center > 0 else 0
                    z_after = z_center + 1 if z_center < len_z-1 else len_z-1
                    if input_channel == 3:
                        batch_x[idx_batch, 1, :, :] = case_x_data[:, :, z_center]
                        batch_x[idx_batch, 0, :, :] = case_x_data[:, :, z_before]
                        batch_x[idx_batch, 2, :, :] = case_x_data[:, :, z_after]
                    if input_channel == 1:
                        batch_x[idx_batch, 0, :, :] = case_x_data[:, :, z_center]
                    if output_channel == 3:
                        batch_y[idx_batch, 0, :, :] = case_y_data[:, :, z_before]
                        batch_y[idx_batch, 1, :, :] = case_y_data[:, :, z_center]
                        batch_y[idx_batch, 2, :, :] = case_y_data[:, :, z_after]
                    if output_channel == 1:
                        batch_y[idx_batch, 0, :, :] = case_y_data[:, :, z_center]

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
                    print("===> Epoch[{:03d}]-Case[{:03d}]({:03d}/{:03d}): ".format(idx_epoch+1, cnt_y+1, idx_iter + 1, len_z//args.batch), end='')
                    print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))

                case_loss[idx_iter] = loss.item()
            
            case_name = os.path.basename(case_x_path)[4:7]
            np.save(args.save_folder+"npy/Epoch[{:03d}]_Case[{}]_t_x.npy".format(idx_epoch+1, case_name), batch_x.cpu().detach().numpy())
            np.save(args.save_folder+"npy/Epoch[{:03d}]_Case[{}]_t_y.npy".format(idx_epoch+1, case_name), batch_y.cpu().detach().numpy())
            np.save(args.save_folder+"npy/Epoch[{:03d}]_Case[{}]_t_z.npy".format(idx_epoch+1, case_name), y_hat.cpu().detach().numpy())

            # after training one case
            loss_mean = np.mean(case_loss)
            loss_std = np.std(case_loss)
            print("===> Epoch[{:03d}]-Case[{:03d}]: ".format(idx_epoch+1, cnt_y+1), end='')
            print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))
            epoch_loss_t[cnt_y] = loss_mean

        # after training all cases
        loss_mean = np.mean(epoch_loss_t)
        loss_std = np.std(epoch_loss_t)
        print("===> Epoch[{}]: ".format(idx_epoch+1), end='')
        print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))
        np.save(args.save_folder+"npy/epoch_loss_t_{:03d}.npy".format(idx_epoch+1), epoch_loss_t)
        train_loss[idx_epoch] = loss_mean
        # ====================================>train<====================================

        # ====================================>val<====================================
        model.eval()
        random.shuffle(list_val_y)
        for cnt_y, path_y in enumerate(list_val_y):

            # train
            case_x_path = path_y.replace("Y", "X")
            case_y_path = path_y
            print("->",case_x_path,"<-", end="")
            case_x_data = nib.load(case_x_path).get_fdata()
            case_y_data = nib.load(case_y_path).get_fdata()
            len_z = case_x_data.shape[2]
            case_loss = np.zeros((len_z//args.batch))
            input_list = list(range(len_z))
            random.shuffle(input_list)

            # 0:[32, 45, 23, 55], 1[76, 74, 54, 99], 3[65, 92, 28, 77], ...
            for idx_iter in range(len_z//args.batch):

                batch_x = np.zeros((args.batch, input_channel, case_x_data.shape[0], case_x_data.shape[1]))
                batch_y = np.zeros((args.batch, output_channel, case_y_data.shape[0], case_y_data.shape[1]))

                for idx_batch in range(args.batch):
                    z_center = input_list[idx_iter*args.batch+idx_batch]
                    z_before = z_center - 1 if z_center > 0 else 0
                    z_after = z_center + 1 if z_center < len_z-1 else len_z-1
                    if input_channel == 3:
                        batch_x[idx_batch, 1, :, :] = case_x_data[:, :, z_center]
                        batch_x[idx_batch, 0, :, :] = case_x_data[:, :, z_before]
                        batch_x[idx_batch, 2, :, :] = case_x_data[:, :, z_after]
                    if input_channel == 1:
                        batch_x[idx_batch, 0, :, :] = case_x_data[:, :, z_center]
                    if output_channel == 3:
                        batch_y[idx_batch, 0, :, :] = case_y_data[:, :, z_before]
                        batch_y[idx_batch, 1, :, :] = case_y_data[:, :, z_center]
                        batch_y[idx_batch, 2, :, :] = case_y_data[:, :, z_after]
                    if output_channel == 1:
                        batch_y[idx_batch, 0, :, :] = case_y_data[:, :, z_center]

                batch_x = torch.from_numpy(batch_x).float().to(device)
                batch_y = torch.from_numpy(batch_y).float().to(device)
                
                y_hat = model(batch_x)
                loss = criterion(y_hat, batch_y)
                case_loss[idx_iter] = loss.item()
            
            # save one progress shot
            case_name = os.path.basename(case_x_path)[4:7]
            np.save(args.save_folder+"npy/Epoch[{:03d}]_Case[{}]_v_x.npy".format(idx_epoch+1, case_name), batch_x.cpu().detach().numpy())
            np.save(args.save_folder+"npy/Epoch[{:03d}]_Case[{}]_v_y.npy".format(idx_epoch+1, case_name), batch_y.cpu().detach().numpy())
            np.save(args.save_folder+"npy/Epoch[{:03d}]_Case[{}]_v_z.npy".format(idx_epoch+1, case_name), y_hat.cpu().detach().numpy())

            # after training one case
            loss_mean = np.mean(case_loss)
            loss_std = np.std(case_loss)
            print("===> Epoch[{:03d}]-Val-Case[{:03d}]: ".format(idx_epoch+1, cnt_y+1), end='')
            print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))
            epoch_loss_v[cnt_y] = loss_mean

        loss_mean = np.mean(epoch_loss_v)
        loss_std = np.std(epoch_loss_v)
        print("===> Epoch[{:03d}]-Val: ".format(idx_epoch+1), end='')
        print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))
        np.save(args.save_folder+"npy/epoch_loss_v_{:03d}.npy".format(idx_epoch+1), epoch_loss_v)
        if loss_mean < best_val_loss:
            # save the best model
            torch.save(model, args.save_folder+"model_best_{:03d}.pth".format(idx_epoch+1))
            print("Checkpoint saved at Epoch {:03d}".format(idx_epoch+1))
            best_val_loss = loss_mean
        # ====================================>val<====================================

    loss_mean = np.mean(train_loss)
    loss_std = np.std(train_loss)
    print("===>Training finished: ", end='')
    print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))

if __name__ == '__main__':
    main()
