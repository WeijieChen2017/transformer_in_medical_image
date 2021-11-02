import os
import glob
import nibabel as nib
import numpy as np

modality_hub = ["CTAC", "InPhase", "OutPhase", "FAT", "NAC", "WATER"]

for modality in modality_hub:
    print("-"*60)
    file_list = sorted(glob.glob("./xue/sub*/*{}.nii.gz".format(modality)))
    max_list = np.zeros(len(file_list))
    min_list = np.zeros(len(file_list))
    cnt = 0
    for file_path in file_list:
        print("--->",os.path.basename(file_path),"<---",end="")
        data = nib.load(file_path).get_fdata()
        max_list[cnt] = np.amax(data)
        min_list[cnt] = np.amin(data)
        print(" Max:", max_list[cnt], "Min: ", min_list[cnt])
        cnt += 1
    print("Max max:", np.amax(max_list), "Min min: ", np.amin(min_list))

