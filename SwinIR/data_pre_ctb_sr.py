from PIL import Image
import nibabel as nib
import numpy as np
import glob
import os

# 3000 for stealth and 1500 for bravo
def normX(data):
    data[data<0] = 0
    data[data>3000] = 3000 
    data = data / 3000
    return data

def normY(data):
    data[data<0] = 0
    data[data>3000] = 3000
    data = data / 3000
    return data

root_folder = "./CTB_SR/"
folderX = root_folder+"CT_128/"
folderY = root_folder+"CT_256/"
valRatio = 0.2
testRatio = 0.1
channelX = 1
channelY = 1

# create directory and search nifty files
trainFolderX = root_folder+"X/train/"
trainFolderY = root_folder+"Y/train/"
testFolderX = root_folder+"X/test/"
testFolderY = root_folder+"Y/test/"
valFolderX = root_folder+"X/val/"
valFolderY = root_folder+"Y/val/"

for folderName in [trainFolderX, testFolderX, valFolderX,
                   trainFolderY, testFolderY, valFolderY]:
    if not os.path.exists(folderName):
        os.makedirs(folderName)

# fileList = glob.glob(folderX+"/mets*.nii") + glob.glob(folderX+"/mets*.nii.gz")
fileList = glob.glob(folderX+"/RSZ*.nii") + glob.glob(folderX+"/RSZ*.nii.gz")
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

# trainList = ['./data_train/NPR_SRC/NPR_051.nii.gz',
#              './data_train/NPR_SRC/NPR_054.nii.gz',
#              './data_train/NPR_SRC/NPR_056.nii.gz',
#              './data_train/NPR_SRC/NPR_057.nii.gz']
# valList = ['./data_train/NPR_SRC/NPR_059.nii.gz']
# testList = ['./data_train/NPR_SRC/NPR_011.nii.gz']

# trainList = ['./data_train/RSPET/RS_051.nii.gz',
#              './data_train/RSPET/RS_054.nii.gz',
#              './data_train/RSPET/RS_056.nii.gz',
#              './data_train/RSPET/RS_057.nii.gz']
# valList = ['./data_train/RSPET/RS_059.nii.gz']
# testList = ['./data_train/RSPET/RS_011.nii.gz']
# trainList = []
# valList = []

# --------------------------------------------------
# Training list:  ['./data_train/NPR_SRC/NPR_001.nii.gz', './data_train/NPR_SRC/NPR_007.nii.gz', './data_train/NPR_SRC/NPR_017.nii.gz', './data_train/NPR_SRC/NPR_019.nii.gz', './data_train/NPR_SRC/NPR_024.nii.gz', './data_train/NPR_SRC/NPR_026.nii.gz', './data_train/NPR_SRC/NPR_028.nii.gz', './data_train/NPR_SRC/NPR_029.nii.gz', './data_train/NPR_SRC/NPR_031.nii.gz', './data_train/NPR_SRC/NPR_044.nii.gz', './data_train/NPR_SRC/NPR_057.nii.gz', './data_train/NPR_SRC/NPR_059.nii.gz', './data_train/NPR_SRC/NPR_067.nii.gz', './data_train/NPR_SRC/NPR_068.nii.gz', './data_train/NPR_SRC/NPR_078.nii.gz', './data_train/NPR_SRC/NPR_082.nii.gz', './data_train/NPR_SRC/NPR_095.nii.gz', './data_train/NPR_SRC/NPR_098.nii.gz', './data_train/NPR_SRC/NPR_101.nii.gz', './data_train/NPR_SRC/NPR_103.nii.gz', './data_train/NPR_SRC/NPR_104.nii.gz', './data_train/NPR_SRC/NPR_130.nii.gz', './data_train/NPR_SRC/NPR_138.nii.gz', './data_train/NPR_SRC/NPR_142.nii.gz', './data_train/NPR_SRC/NPR_159.nii.gz']
# --------------------------------------------------
# Validation list:  ['./data_train/NPR_SRC/NPR_051.nii.gz', './data_train/NPR_SRC/NPR_054.nii.gz', './data_train/NPR_SRC/NPR_056.nii.gz', './data_train/NPR_SRC/NPR_097.nii.gz', './data_train/NPR_SRC/NPR_127.nii.gz', './data_train/NPR_SRC/NPR_128.nii.gz', './data_train/NPR_SRC/NPR_133.nii.gz']
# --------------------------------------------------
# Testing list:  ['./data_train/NPR_SRC/NPR_011.nii.gz', './data_train/NPR_SRC/NPR_063.nii.gz', './data_train/NPR_SRC/NPR_143.nii.gz']
# --------------------------------------------------


print('-'*50)
print("Training list: ", trainList)
print('-'*50)
print("Validation list: ", valList)
print('-'*50)
print("Testing list: ", testList)
print('-'*50)

packageTrain = [trainList, trainFolderX, trainFolderY, "Train"]
packageVal = [valList, valFolderX, valFolderY, "Validation"]
packageTest = [testList, testFolderX, testFolderY, "Test"]
np.save(root_folder+"dataset_division.npy", [packageTrain, packageVal, packageTest])

for package in [packageVal, packageTrain, packageTest]: # 

    fileList = package[0]
    folderX = package[1]
    folderY = package[2]
    print("-"*25, package[3], "-"*25)

    # npy version
    for pathX in fileList:

        print(pathX)
        pathY = pathX.replace("128", "256")
        filenameX = os.path.basename(pathX)[4:7]
        filenameY = os.path.basename(pathY)[4:7]
        dataX = nib.load(pathX).get_fdata()
        dataY = nib.load(pathY).get_fdata()
        dataNormX = normX(dataX)
        dataNormY = normY(dataY)

        np.save(folderX + "RSZ_" + filenameX + ".npy", dataNormX)
        np.save(folderY + "RSZ_" + filenameY + ".npy", dataNormY)        
        print(folderX + "RSZ_" + filenameX + ".npy")
    print(len(fileList), " files are saved. ")
