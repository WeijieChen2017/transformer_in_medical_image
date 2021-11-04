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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default="7", help='Use which GPU to train')
    parser.add_argument('--folder_X_te', type=str, default="./MR2CT_B_SWINIR/pred/", help='input folder of T1MAP PET images')
    parser.add_argument('--folder_Y_te', type=str, default="./MR2CT_B_SWINIR/Y/test/", help='input folder of BRAVO images')
    parser.add_argument('--weights_path', type=str, default='./CTB_SR_best_model/CTB_SR_model_best_015.pth')
    args = parser.parse_args()

    gpu_list = ','.join(str(x) for x in args.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    device = torch.device('cuda' if  torch.cuda.is_available() else 'cpu')

    for path in ["./MR2CT_B_SWINIR/pred_SR/"]:
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

        cube_x_path = X_path
        cube_y_path = args.folder_Y_te+"RSZ_"+os.path.basename(X_path)[4:7]+".npy"
        print("->",cube_x_path, "<->", cube_y_path, "<-",end="")
        # cube_x_data = np.load(cube_x_path)
        cube_y_data = np.load(cube_y_path)
        cube_x_data = nib.load(cube_x_path).get_fdata()
        # cube_y_data = nib.load(cube_y_path).get_fdata()
        len_z = cube_x_data.shape[2]
        y_hat = np.zeros(cube_y_data.shape)
        
        for idx in range(len_z):

            batch_x = np.zeros((1, 3, cube_x_data.shape[0], cube_x_data.shape[1]))
            batch_y = cube_y_data[:, idx, :]

            z_center = idx
            batch_x[0, 1, :, :] = cube_x_data[:, :, z_center]
            z_before = z_center - 1 if z_center > 0 else 0
            z_after = z_center + 1 if z_center < len_z-1 else len_z-1
            batch_x[0, 0, :, :] = cube_x_data[:, :, z_before]
            batch_x[0, 2, :, :] = cube_x_data[:, :, z_after]

            batch_x = torch.from_numpy(batch_x).float().to(device)

            y_hat_output = model(batch_x).cpu().detach().numpy()
            y_hat[:, :, idx] = np.squeeze(y_hat_output[:, 1, :, :])
        
        for cnt_loss, loss_fnc in enumerate(criterion_list):
            curr_loss = loss_fnc(cube_y_data, y_hat).item()
            loss_mat[cnt_X, cnt_loss] = curr_loss
            print("===> Loss[{}]: {:6}".format(loss_fnc.__name__, curr_loss), end='')
        
        nifty_name = "./MR2CT/ct_bravo/CT__MLAC_" + os.path.basename(X_path)[5:7]+"_MNI.nii.gz"
        nifty_file = nib.load(nifty_name)
        print("Loaded from", nifty_name, end="")


        pred_file = nib.Nifti1Image(y_hat, nifty_file.affine, nifty_file.header)
        pred_name = "./MR2CT_B_SWINIR/pred_SR/"+"PRD_"+os.path.basename(X_path)[4:7]+".nii.gz"
        nib.save(pred_file, pred_name)
        print(" Saved to", pred_name)

if __name__ == '__main__':
    main()
