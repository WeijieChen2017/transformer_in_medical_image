import os
import glob

for idx in range(100):
    cmd = "3dresample -dxyz 2 2 1 -rmode Cu -prefix 128_{:03d}.nii.gz ".format(idx)
    cmd += "-input CT__MLAC_{:02d}".format(idx)+"_MNI.nii.gz" 
    print(cmd)
    os.system(cmd)