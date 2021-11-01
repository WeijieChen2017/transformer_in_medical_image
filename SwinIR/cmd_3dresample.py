import os
import glob

for idx in range(64):
    cmd = "3dresample -dxyz 0.667 0.667 1 -rmode Cu -prefix 384_{:03d}.nii.gz ".format(idx)
    cmd += "-input MR__MLAC_{:02d}".format(idx)+"_MNI.nii.gz" 
    print(cmd)
    os.system(cmd)