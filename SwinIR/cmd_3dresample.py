import os
import glob

for idx in range(64):
    cmd = "3dresample -dxyz 0.875 0.875 1 -rmode Cu -prefix 224_{:03d}.nii.gz ".format(idx)
    cmd += "-input CT__MLAC_{:02d}".format(idx)+"_MNI.nii.gz" 
    print(cmd)
    os.system(cmd)