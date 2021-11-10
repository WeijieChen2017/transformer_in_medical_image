import os
import glob

for idx in range(100):
    cmd = "3dresample -dxyz 0.8594 0.8594 1 -rmode Cu -prefix 256_{:03d}.nii.gz ".format(idx)
    cmd += "-input CT__MLAC_{:02d}".format(idx)+"_ORIG.nii.gz" 
    print(cmd)
    os.system(cmd)