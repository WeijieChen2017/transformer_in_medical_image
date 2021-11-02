import os
import cv2
import glob
import torch
import random
import argparse
import numpy as np
import nibabel as nib
import torch.nn as nn

# from models.network_swinir import SwinIR as net
from unet import UNet
np.random.seed(seed=813)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_channel', type=int, default=5, help='the number of input channel')
    parser.add_argument('--output_channel', type=int, default=1, help='the number of output channel')
    parser.add_argument('--save_folder', type=str, default="./xue_5to1/", help='Save_prefix')
    parser.add_argument('--gpu_ids', type=str, default="7", help='Use which GPU to train')
    parser.add_argument('--epoch', type=int, default=900, help='how many epochs to train')
    parser.add_argument('--batch', type=int, default=5, help='how many batches in one run')
    parser.add_argument('--loss_display_per_iter', type=int, default=600, help='display how many losses per iteration')
    parser.add_argument('--folder_train', type=str, default="./xue/train/", help='input folder of trianing data X')
    parser.add_argument('--folder_val', type=str, default="./xue/val/", help='input folder of validation data X')
    parser.add_argument('--weights_path', type=str, default='./pretrain_models/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth')
    
    args = parser.parse_args()
    input_channel = args.input_channel
    output_channel = args.output_channel

    gpu_list = ','.join(str(x) for x in args.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for path in [args.save_folder, args.save_folder+"npy/"]:
        if not os.path.exists(path):
            os.mkdir(path)

    model = UNet(n_channels=input_channel, n_classes=output_channel, bilinear=True)
    model.train().float()
    model = model.to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    list_train = sorted(glob.glob(args.folder_train+"*_NAC.nii.gz"))
    list_val = sorted(glob.glob(args.folder_val+"*_NAC.nii.gz"))

    train_loss = np.zeros((args.epoch)) # over the whole process
    epoch_loss_t = np.zeros((len(list_train))) # over the training part of each epoch
    epoch_loss_v = np.zeros((len(list_val))) # over the validation part of each epoch
    best_val_loss = 1e6 # save the best model if the validation loss is less than this
    per_iter_loss = np.zeros((args.loss_display_per_iter)) # to show loss every X batch inside one case
    case_loss = None # the loss over one case

    for idx_epoch in range(args.epoch):
        print("~~~~~~Epoch[{:03d}]~~~~~~".format(idx_epoch+1))

        # ====================================>train<====================================
        model.train()
        # randonmize training dataset
        random.shuffle(list_train)
        for cnt, path in enumerate(list_train):

            case_nac_path = path
            print("->",case_nac_path,"<-", end="")
            case_name = os.path.basename(case_nac_path)[5:8]
            case_nac_data = nib.load(case_nac_path).get_fdata()
            case_sct_data = nib.load(case_nac_path.replace("NAC", "SCT")).get_fdata()
            case_inp_data = nib.load(case_nac_path.replace("NAC", "INP")).get_fdata()
            case_oup_data = nib.load(case_nac_path.replace("NAC", "OUP")).get_fdata()
            case_fat_data = nib.load(case_nac_path.replace("NAC", "FAT")).get_fdata()
            case_wat_data = nib.load(case_nac_path.replace("NAC", "WAT")).get_fdata()
            len_z = case_nac_data.shape[2]
            case_loss = np.zeros((len_z//args.batch))
            input_list = list(range(len_z))
            random.shuffle(input_list)

            # 0:[32, 45, 23, 55], 1[76, 74, 54, 99], 3[65, 92, 28, 77], ...
            for idx_iter in range(len_z//args.batch):

                batch_x = np.zeros((args.batch, input_channel, case_nac_data.shape[0], case_nac_data.shape[1]))
                batch_y = np.zeros((args.batch, output_channel, case_nac_data.shape[0], case_nac_data.shape[1]))

                for idx_batch in range(args.batch):
                    z_center = input_list[idx_iter*args.batch+idx_batch]
                    batch_x[idx_batch, 0, :, :] = case_inp_data[:, :, z_center]
                    batch_x[idx_batch, 1, :, :] = case_oup_data[:, :, z_center]
                    batch_x[idx_batch, 2, :, :] = case_nac_data[:, :, z_center]
                    batch_x[idx_batch, 3, :, :] = case_wat_data[:, :, z_center]
                    batch_x[idx_batch, 4, :, :] = case_fat_data[:, :, z_center]
                    
                    batch_y[idx_batch, 0, :, :] = case_sct_data[:, :, z_center]

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
                    print("===> Epoch[{:03d}]-Case[{}]({:03d}/{:03d}): ".format(idx_epoch+1, case_name, idx_iter + 1, len_z//args.batch), end='')
                    print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))

                case_loss[idx_iter] = loss.item()
            
            # np.save(args.save_folder+"npy/Epoch[{:03d}]_Case[{}]_t_x.npy".format(idx_epoch+1, case_name), batch_x.cpu().detach().numpy())
            # np.save(args.save_folder+"npy/Epoch[{:03d}]_Case[{}]_t_y.npy".format(idx_epoch+1, case_name), batch_y.cpu().detach().numpy())
            # np.save(args.save_folder+"npy/Epoch[{:03d}]_Case[{}]_t_z.npy".format(idx_epoch+1, case_name), y_hat.cpu().detach().numpy())

            # after training one case
            loss_mean = np.mean(case_loss)
            loss_std = np.std(case_loss)
            print("===> Epoch[{:03d}]-Case[{:03d}]: ".format(idx_epoch+1, cnt+1), end='')
            print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))
            epoch_loss_t[cnt] = loss_mean

        # after training all cases
        loss_mean = np.mean(epoch_loss_t)
        loss_std = np.std(epoch_loss_t)
        print("===> Epoch[{}]: ".format(idx_epoch+1), end='')
        print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))
        np.save(args.save_folder+"npy/epoch_loss_t_{:03d}.npy".format(idx_epoch+1), epoch_loss_t)
        train_loss[idx_epoch] = loss_mean
        torch.cuda.empty_cache()
        # ====================================>train<====================================

        # ====================================>val<====================================
        model.eval()
        random.shuffle(list_val)
        for cnt, path in enumerate(list_val):

            case_nac_path = path
            print("->",case_nac_path,"<-", end="")
            case_name = os.path.basename(case_nac_path)[5:8]
            case_nac_data = nib.load(case_nac_path).get_fdata()
            case_sct_data = nib.load(case_nac_path.replace("NAC", "SCT")).get_fdata()
            case_inp_data = nib.load(case_nac_path.replace("NAC", "INP")).get_fdata()
            case_oup_data = nib.load(case_nac_path.replace("NAC", "OUP")).get_fdata()
            case_fat_data = nib.load(case_nac_path.replace("NAC", "FAT")).get_fdata()
            case_wat_data = nib.load(case_nac_path.replace("NAC", "WAT")).get_fdata()
            len_z = case_nac_data.shape[2]
            case_loss = np.zeros((len_z//args.batch))
            input_list = list(range(len_z))
            random.shuffle(input_list)

            # 0:[32, 45, 23, 55], 1[76, 74, 54, 99], 3[65, 92, 28, 77], ...
            for idx_iter in range(len_z//args.batch):

                batch_x = np.zeros((args.batch, input_channel, case_nac_data.shape[0], case_nac_data.shape[1]))
                batch_y = np.zeros((args.batch, output_channel, case_nac_data.shape[0], case_nac_data.shape[1]))

                for idx_batch in range(args.batch):
                    z_center = input_list[idx_iter*args.batch+idx_batch]
                    batch_x[idx_batch, 0, :, :] = case_inp_data[:, :, z_center]
                    batch_x[idx_batch, 1, :, :] = case_oup_data[:, :, z_center]
                    batch_x[idx_batch, 2, :, :] = case_nac_data[:, :, z_center]
                    batch_x[idx_batch, 3, :, :] = case_wat_data[:, :, z_center]
                    batch_x[idx_batch, 4, :, :] = case_fat_data[:, :, z_center]
                    
                    batch_y[idx_batch, 0, :, :] = case_sct_data[:, :, z_center]
                batch_x = torch.from_numpy(batch_x).float().to(device)
                batch_y = torch.from_numpy(batch_y).float().to(device)
                
                y_hat = model(batch_x)
                loss = criterion(y_hat, batch_y)
                case_loss[idx_iter] = loss.item()
            
            # save one progress shot
            
            np.save(args.save_folder+"npy/Epoch[{:03d}]_Case[{}]_v_x.npy".format(idx_epoch+1, case_name), batch_x.cpu().detach().numpy())
            np.save(args.save_folder+"npy/Epoch[{:03d}]_Case[{}]_v_y.npy".format(idx_epoch+1, case_name), batch_y.cpu().detach().numpy())
            np.save(args.save_folder+"npy/Epoch[{:03d}]_Case[{}]_v_z.npy".format(idx_epoch+1, case_name), y_hat.cpu().detach().numpy())

            # after training one case
            loss_mean = np.mean(case_loss)
            loss_std = np.std(case_loss)
            print("===> Epoch[{:03d}]-Val-Case[{}]: ".format(idx_epoch+1, case_name), end='')
            print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))
            epoch_loss_v[cnt] = loss_mean

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
        torch.save(model, args.save_folder+"model_latest.pth")
        torch.cuda.empty_cache()
        # ====================================>val<====================================

    loss_mean = np.mean(train_loss)
    loss_std = np.std(train_loss)
    print("===>Training finished: ", end='')
    print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))

if __name__ == '__main__':
    main()
