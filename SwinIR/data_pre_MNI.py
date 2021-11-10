import nibabel as nib
import numpy as np
import glob
import os

# 3000 for stealth and 1500 for bravo
# X / 2000
# Y [-1000, 2000] / 3000

def normX(data):
    data[data<0] = 0
    data[data>2000] = 2000 
    data = data / 2000
    return data

def normY(data):
    data[data<-1000] = -1000
    data[data>2000] = 2000
    data = data / 3000
    return data

def normY_offset1000(data):
    data[data<0] = 0
    data[data>3000] = 3000
    data = data / 3000
    return data

root_folder = "./MR2CT/"
save_folder = "./MR2CT_B_SWINIR_MNI_256/"
search_folderX = root_folder+"MNI/"
search_folderY = root_folder+"MNI/"
valRatio = 0.2
testRatio = 0.1
channelX = 1
channelY = 1

# create directory and search nifty files
trainFolderX = save_folder+"X/train/"
trainFolderY = save_folder+"Y/train/"
testFolderX = save_folder+"X/test/"
testFolderY = save_folder+"Y/test/"
valFolderX = save_folder+"X/val/"
valFolderY = save_folder+"Y/val/"

for folderName in [trainFolderX, testFolderX, valFolderX,
                   trainFolderY, testFolderY, valFolderY]:
    if not os.path.exists(folderName):
        os.makedirs(folderName)

# fileList = glob.glob(folderX+"/mets*.nii") + glob.glob(folderX+"/mets*.nii.gz")
fileList = glob.glob(search_folderX+"/MR*.nii") + glob.glob(search_folderX+"/MR*.nii.gz")
fileList.sort()
for filePath in fileList:
    print(filePath)

# shuffle and create train/val/test file list
np.random.seed(813)
fileList = np.asarray(fileList)
np.random.shuffle(fileList)
fileList = list(fileList)

testList = sorted(fileList)
valList = []
trainList = []

# valList = fileList[:int(len(fileList)*valRatio)]
# valList.sort()
# testList = fileList[-int(len(fileList)*testRatio):]
# testList.sort()
# trainList = list(set(fileList) - set(valList) - set(testList))
# trainList.sort()

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
        pathY = search_folderY+os.path.basename(pathX).replace("MR", "CT")
        filenameX = os.path.basename(pathX)[9:11]
        filenameY = os.path.basename(pathY)[9:11]
        dataX = nib.load(pathX).get_fdata()
        dataY = nib.load(pathY).get_fdata()
        print("X:", np.amax(dataX), np.amin(dataX), "<--> Y:", np.amax(dataY), np.amin(dataY))
        dataNormX = normX(dataX)
        if filenameX in ["04", "23", "49", "59"]:
            print("Special norm!")
            dataNormY = normY_offset1000(dataY)
        else:
            dataNormY = normY(dataY)
        print(dataNormX.shape, dataNormY.shape)

        np.save(folderX + "RSZ_0" + filenameX + ".npy", dataNormX)
        np.save(folderY + "RSZ_0" + filenameY + ".npy", dataNormY)        
        print(folderX + "RSZ_0" + filenameX + ".npy")
    print(len(fileList), " files are saved. ")
