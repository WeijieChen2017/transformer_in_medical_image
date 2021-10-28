# python main_test_swinir.py --task classical_sr --scale 2 --training_patch_size 64 --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth --folder_lq testsets/Set5/LR_bicubic/X2 --folder_gt testsets/Set5/HR

# 3dresample -dxyz 2.734 2.734 1.367 -prefix RSZ_2x.nii.gz -input CUB_011.nii.gz
# 3dresample -dxyz 5.468 5.468 1.367 -prefix RSZ_4x.nii.gz -input CUB_011.nii.gz
# 3dresample -dxyz 10.936 10.936 1.367 -prefix RSZ_8x.nii.gz -input CUB_011.nii.gz
# 3dresample -dxyz 0.664 0.664 3 -prefix RSZ_PET.nii.gz -input MAC_PET.nii.gz
# 3dresample -dxyz 0.332 0.332 3 -prefix RSZ_4x_PET.nii.gz -input MAC_PET.nii.gz
# 3dresample -dxyz 0.4688 0.4688 0.6 -prefix brain_4x.nii.gz -input brain1_pet.nii.gz
# 3dresample -dxyz 1.8752 1.8752 0.6 -prefix brain_1x.nii.gz -input brain_4x.nii.gz
# 3dresample -dxyz 0.1172 0.1172 0.6 -prefix brain_16x.nii.gz -input brain_4x.nii.gz

import nibabel as nib
import numpy as np
import time
import os

def normY(data):
    data[data<-1000] = -1000
    data[data>3000] = 3000
    data = (data + 1000) / 4000
    return data

def normX(data):
    print(np.amax(data))
    return data / np.amax(data)

def get_index(current_idx, max_idx):
    if current_idx == 0:
        return [0, 0, 1]
    if current_idx == max_idx:
        return [max_idx-1, max_idx, max_idx]
    else:
        return [current_idx-1, current_idx, current_idx+1]

def volume2slice(data, save_folder):
    dx, dy, dz = data.shape
    img = np.zeros((dx, dy, 3))
    for idx in range(dz):
        idx_set = get_index(idx, dz-1)
        img[:, :, 0] = data[:, :, idx_set[0]]
        img[:, :, 1] = data[:, :, idx_set[1]]
        img[:, :, 2] = data[:, :, idx_set[2]]
        np.save(save_folder+"brain_{:03d}.npy".format(idx), img)
        print("Save imgs in "+save_folder+" [{:03d}]/[{:03d}]".format(idx+1, dz+1))


# (imgname, imgext) = os.path.splitext(os.path.basename(path))
# img_gt = np.load(path)
# img_lq = np.load(f'{args.folder_lq}/{imgname}x{args.scale}{imgext}')

nifty_list = []
for filename in ["./brain/brain_4x.nii.gz"]:
    nifty_file = nib.load(filename)
    nifty_data = nifty_file.get_fdata()
    print("header: ", nifty_file.header)
    nifty_list.append(normX(nifty_data))
    print(nifty_data.shape)
print("======>Data is loaded.<======")
time.sleep(5)

save_list = ["./test/brain/HR/"]
for save_dir in save_list:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

for idx in range(len(nifty_list)):
    volume2slice(nifty_list[idx], save_list[idx])

# cmd = "python main_test_swinir.py "
# cmd += "--task classical_sr --scale 2 --training_patch_size 64 "
# cmd += "--model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth "
# cmd += "--folder_lq ./test/CT/LR/ "
# cmd += "--folder_gt ./test/CT/HR/"
cmd = "python main_test_swinir.py --task real_sr --scale 4 --large_model --model_path model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth --folder_lq ./test/brain/HR/"
print(cmd)
# os.system(cmd)

# python main_test_swinir.py --task classical_sr --scale 2 --training_patch_size 64 --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth --folder_lq ./test/PET/LR/ --folder_gt ./test/CT/HR/
# python main_test_swinir.py --task classical_sr --scale 2 --training_patch_size 64 --model_path model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth --folder_lq ./test/CT/LR_2x/ --folder_gt ./test/CT/HR/
# python main_test_swinir.py --task real_sr --scale 4 --large_model --model_path model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth --folder_lq ./test/PET/LR/

