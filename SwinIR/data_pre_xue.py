from PIL import Image
import nibabel as nib
import numpy as np
import glob
import os

# [ CTAC ] Max max: 1322.656982421875 Min min:  -1000.0
# [ InPhase ] Max max: 4376.7490234375 Min min:  0.0
# [ OutPhase ] Max max: 4193.5478515625 Min min:  0.0
# [ FAT ] Max max: 4270.2763671875 Min min:  0.0
# [ NAC ] Max max: 11732.232421875 Min min:  0.0
# [ WATER ] Max max: 2838.23681640625 Min min:  0.0
# SCT INP OUP FAT NAC WAT

def normSCT(data):
    data[data<-1000] = -1000
    data[data>1500] = 1500
    data = data + 1000
    data = data / 2500
    return data

def normINP(data):
    return data/5000

def normOUP(data):
    return data/5000

def normFAT(data):
    return data/5000

def normNAC(data):
    return data/1500

def normWAT(data):
    return data/3000



root_folder = "./xue/"
save_folder = "./xue/"
# search_folderX = root_folder+"t1_bravo/"
# search_folderY = root_folder+"ct_bravo/"
valRatio = 0.2
testRatio = 0.1
channelX = 1
channelY = 1

# create directory and search nifty files
trainFolderX = save_folder+"train/"
# trainFolderY = save_folder+"train/"
testFolderX = save_folder+"test/"
# testFolderY = save_folder+"test/"
valFolderX = save_folder+"val/"
# valFolderY = save_folder+"val/"

for folderName in [trainFolderX, testFolderX, valFolderX]:
                   # trainFolderY, testFolderY, valFolderY]:
    if not os.path.exists(folderName):
        os.makedirs(folderName)

# fileList = glob.glob(folderX+"/mets*.nii") + glob.glob(folderX+"/mets*.nii.gz")
fileList = glob.glob("./xue/*/*NAC.nii.gz")
fileList.sort()
for filePath in fileList:
    print(filePath)

# shuffle and create train/val/test file list
np.random.seed(813)
fileList = np.asarray(fileList)
np.random.shuffle(fileList)
fileList = list(fileList)

valList = fileList[:int(len(fileList)*valRatio)]
valList.sort()
testList = fileList[-int(len(fileList)*testRatio):]
testList.sort()
trainList = list(set(fileList) - set(valList) - set(testList))
trainList.sort()

print('-'*50)
print("Training list: ", trainList)
print('-'*50)
print("Validation list: ", valList)
print('-'*50)
print("Testing list: ", testList)
print('-'*50)

packageTrain = [trainList, trainFolderX, "Train"]
packageVal = [valList, valFolderX, "Validation"]
packageTest = [testList, testFolderX, "Test"]
np.save(root_folder+"dataset_division.npy", [packageTrain, packageVal, packageTest])

for package in [packageVal, packageTrain, packageTest]: # 

    fileList = package[0]
    folder = package[1]
    # folderY = package[2]
    print("-"*25, package[2], "-"*25)

    # SCT INP OUP FAT NAC WAT
    for pathX in fileList:

        print(pathX)
        case_number = os.path.basename(pathX)[4:7]
        fileX = nib.load(pathX)
        dataX = fileX.get_fdata()
        dataNormX = normNAC(dataX)
        fileNormX = nib.Nifti1Image(dataNormX, fileX.affine, fileX.header)
        nameX = folder + "NORM_" + case_number + "_NAC.nii.gz"
        nib.save(fileNormX, nameX)
        print("Saved to", nameX)
        
        replacements = ["CTAC", "InPhase", "OutPhase", "FAT", "WATER"]
        norm = [normSCT, normINP, normOUP, normFAT, normWAT]
        cnt = 0
        for modality in ["SCT", "INP", "OUP", "FAT", "WAT"]:
            path = pathX.replace("NAC", replacements[cnt])
            file = nib.load(path)
            data = file.get_fdata()
            dataNorm = norm[cnt](data)
            fileNorm = nib.Nifti1Image(dataNorm, file.affine, file.header)
            name = folder + "NORM_" + case_number + "_{}.nii.gz".format(modality)
            nib.save(fileNorm, name)
            print("Saved to", name)
            cnt += 1
    print(len(fileList), " cases are saved. ")
