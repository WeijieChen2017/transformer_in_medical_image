import nibabel as nib
import numpy as np
import glob
import os

# t1 5000 t2 3000
def normX(data):
    data[data<0] = 0
    data[data>5000] = 5000 
    data = data / 5000
    return data

def normY(data):
    data[data<0] = 0
    data[data>4000] = 4000
    data = data / 4000
    return data

root_folder = "./large_CT/"
save_folder = "./deepMRAC_CT/"
search_folderX = root_folder
# search_folderX = root_folder+"t1_bravo/"
# search_folderY = root_folder+"ct_bravo/"
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
fileList = glob.glob(search_folderX+"/*.nii.gz")
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
        # pathY = pathX.replace("t1", "t2")
        filenameX = os.path.basename(pathX)[5:8]
        # filenameY = os.path.basename(pathY)[11:15]
        fileX = nib.load(pathX)
        dataX = fileX.get_fdata()
        # dataY = nib.load(pathY).get_fdata()
        dataNormX = normY(dataX)
        # dataNormY = normY(dataY)
        # print(dataNormX.shape, dataNormY.shape)

        fileNormX = nib.Nifti1Image(dataNormX, fileX.affine, fileX.header)
        nameX = folderX + "RSZ_" + filenameX + ".nii.gz"
        nib.save(fileNormX, nameX)
        print("Saved to", nameX)

    print(len(fileList), " files are saved. ")
