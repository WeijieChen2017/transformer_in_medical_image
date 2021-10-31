import os
for idx in range(10):
    name_sct = "CT__MLAC_{:01d}_MNI.nii.gz".format(idx)
    name_mri = "MR__MLAC_{:01d}_MNI.nii.gz".format(idx)
    name_sct_new = "CT__MLAC_{:02d}_MNI.nii.gz".format(idx)
    name_mri_new = "MR__MLAC_{:02d}_MNI.nii.gz".format(idx)
    cmd_mri = "mv ./MR2CT/t1_stealth/"+name_mri+" ./MR2CT/t1_stealth/"+name_mri_new
    cmd_sct = "mv ./MR2CT/ct_stealth/"+name_sct+" ./MR2CT/ct_stealth/"+name_sct_new
    print(cmd_mri)
    os.system(cmd_mri)
    print(cmd_sct)
    os.system(cmd_sct)
