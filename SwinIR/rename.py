import os
for idx in range(10):
    name_sct = "CT__MLAC_"+str(idx)+"_MNI.nii.gz"
    name_mri = "MR__MLAC_"+str(idx)+"_MNI.nii.gz"
    name_sct_new = "CT__MLAC_{:02d}+_MNI.nii.gz".format(idx)
    name_mri_new = "MR__MLAC_{:02d}+_MNI.nii.gz".format(idx)
    cmd_mri = "mv ./MR2CT/t1_bravo/"+name_mri+" ./MR2CT/t1_bravo/"+name_mri_new
    cmd_sct = "mv ./MR2CT/ct_bravo/"+name_sct+" ./MR2CT/ct_bravo/"+name_sct_new
    print(cmd_mri)
    os.system(cmd_mri)
    print(cmd_sct)
    os.system(cmd_sct)
