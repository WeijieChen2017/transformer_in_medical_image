import os
import glob

for idx in [11, 63, 143]:
    cmd = "3dresample -dxyz 2.734 2.734 5.468 -rmode Cu -prefix 256_sCT_{:03d}.nii.gz ".format(idx)
    cmd += "-input sCT_CUB_{:03d}".format(idx)+".nii.gz" 
    print(cmd)
    os.system(cmd)


# 3dresample -dxyz 1 1 1 -rmode Cu -prefix RSZ_011.nii.gz -input CUB_011.nii.gz