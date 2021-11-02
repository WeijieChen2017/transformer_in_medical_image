import argparse
import cv2
import glob
import time
import random
import numpy as np
import torch.nn as nn
import nibabel as nib
from collections import OrderedDict
from sys import getsizeof
import os
import torch
import requests

from models.network_swinir import SwinIR as net
from utils import util_calculate_psnr_ssim as util

np.random.seed(seed=813)

def denormY(data):
    data = data * 3000
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default="7", help='Use which GPU to train')
    parser.add_argument('--folder_X_te', type=str, default="./xue/test/", help='input folder of T1MAP PET images')
    parser.add_argument('--weights_path', type=str, default='./xue_5to1/model_best_067.pth')
    parser.add_argument('--save_folder', type=str, default="./xue_5to1/", help='Save_prefix')
    
    args = parser.parse_args()

    gpu_list = ','.join(str(x) for x in args.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')

    for path in [args.save_folder+"pred/"]:
        if not os.path.exists(path):
            os.mkdir(path)

    print(f'loading model from {args.weights_path}')
    model = torch.load(args.weights_path)
    model.eval().float()
    model = model.to(device)
    
    X_list = sorted(glob.glob(args.folder_X_te+"*.nii.gz"))
    # criterion_list = [nn.L1Loss, nn.MSELoss, nn.SmoothL1Loss]
    criterion_list = []
    # (nii_file, loss)
    loss_mat = np.zeros((len(X_list), len(criterion_list)))

    for cnt_X, X_path in enumerate(X_list):

        case_nac_path = X_path
        print("->",case_nac_path,"<-", end="")
        case_name = os.path.basename(case_nac_path)[5:8]
        case_nac_data = nib.load(case_nac_path).get_fdata()
        case_sct_data = nib.load(case_nac_path.replace("NAC", "SCT")).get_fdata()
        case_inp_data = nib.load(case_nac_path.replace("NAC", "INP")).get_fdata()
        case_oup_data = nib.load(case_nac_path.replace("NAC", "OUP")).get_fdata()
        case_fat_data = nib.load(case_nac_path.replace("NAC", "FAT")).get_fdata()
        case_wat_data = nib.load(case_nac_path.replace("NAC", "WAT")).get_fdata()
        len_z = case_nac_data.shape[2]
        y_hat = np.zeros(case_sct_data.shape)

        for idx in range(len_z):

            batch_x = np.zeros((1, 5, case_nac_data.shape[0], case_nac_data.shape[1]))
            
            z_center = idx
            batch_x[0, 0, :, :] = case_inp_data[:, :, z_center]
            batch_x[0, 1, :, :] = case_oup_data[:, :, z_center]
            batch_x[0, 2, :, :] = case_nac_data[:, :, z_center]
            batch_x[0, 3, :, :] = case_wat_data[:, :, z_center]
            batch_x[0, 4, :, :] = case_fat_data[:, :, z_center]

            batch_x = torch.from_numpy(batch_x).float().to(device)

            y_hat_output = model(batch_x).cpu().detach().numpy()
            y_hat[:, :, idx] = np.squeeze(y_hat_output)
        
        for cnt_loss, loss_fnc in enumerate(criterion_list):
            curr_loss = loss_fnc(cube_y_data, y_hat).item()
            loss_mat[cnt_X, cnt_loss] = curr_loss
            print("===> Loss[{}]: {:6}".format(loss_fnc.__name__, curr_loss), end='')
        
        nifty_name = nib.load(case_nac_path.replace("NAC", "SCT"))
        nifty_file = nib.load(nifty_name)
        print("Loaded from", nifty_name, end="")

        pred_file = nib.Nifti1Image(denormY(y_hat), nifty_file.affine, nifty_file.header)
        pred_name = args.save_folder+"pred/"+"PRD_"+os.path.basename(X_path)[5:8]+".nii.gz"
        nib.save(pred_file, pred_name)
        print(" Saved to", pred_name)

if __name__ == '__main__':
    main()
