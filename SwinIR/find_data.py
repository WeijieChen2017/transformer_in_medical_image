import os
import glob

file_list = sorted(glob.glob("/shares/petmr/deepMRAC_direct/nifti/*/ctac/ctac.nii"))
for file_path in file_list:
    # print(file_path)
    path_split = file_path.split("/")
    filename = path_split[5][-3:]
    # print(filename)
    cmd = "cp "+file_path+" ./CTAC_"+filename+".nii"
    print(cmd)
    os.system(cmd)