import os
import glob
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

model_list = sorted(glob.glob("./bridge_3000/*/*.pth"))
work_dir = "./bridge_3000/weights_analysis/"
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

model_hub = {}

for model_path in model_list:
    model_name = model_path.split("/")[-2]
    model_best = int(os.path.basename(model_path)[-7:-4])
    if not model_name in model_hub.keys():
        model_hub[model_name] = model_best
    else:
        if model_best > model_hub[model_name]:
            model_hub[model_name] = model_best

print(model_hub)
# {'CT': 50, 'CT_50': 4, 'CT_aug': 41, 'CT_intra': 34, 
# 'CT_intra_aug': 44, 'CT_skip': 50, 'MR-tf6-CT': 50, 
# 'MR-tf6-CT_50': 17, 'MR-tf6-CT_from_naive_skip': 39, 
# 'MR': 28, 'MR_intra': 23, 'MR_skip': 22, 'large_CT': 29, 
# 'naive-tf': 41, 'naive': 43, 'naive_aug': 46, 'naive_intra': 70, 
# 'naive_intra_aug': 46, 'naive_skip': 44, 'naive_tf_skip_aug': 45}


for model_name in model_hub:

    model_best = model_hub[model_name]
    # print(model_name, model_best)
# model_name = "CT"
# model_best = 50
    model_path = "./bridge_3000/"+model_name+"/model_best_{:03d}.pth".format(model_best)
    model = torch.load(model_path)
    model_weights = model.state_dict()
    layer_hub = model_weights.keys()
    # >>> print(type(model_weights))
    # <class 'collections.OrderedDict'>

    # 'up4.conv.double_conv.1.weight'
    # 'up4.conv.double_conv.1.bias'
    # 'up4.conv.double_conv.1.running_mean'
    # 'up4.conv.double_conv.1.running_var'
    # 'up4.conv.double_conv.1.num_batches_tracked'

    module_hub = []
    for elem in model_weights:
        module = elem.split(".")[0]
        if not module in module_hub:
            module_hub.append(module)
    # print(module_hub)

    target_hub = []
    axis_y = 0

    for module in module_hub:
    # module = "down1"
        sub_module_hub = []
        for elem in model_weights:
            elem_module = elem.split(".")[0]
            if elem_module == module:
                sub_module_hub.append(elem)
        # print(sub_module_hub)
        for target_weight in ["weight", "bias"]:
            for elem in sub_module_hub:
                if target_weight in elem and len(model_weights[elem].shape[0]) > 1:
                    target_hub.append(elem)
                    # print(elem, model_weights[elem].size())
                    if model_weights[elem].shape[0] > axis_y:
                        axis_y = model_weights[elem].shape[0]

    axis_x = len(target_hub)

    #down1.maxpool_conv.1.double_conv.0.weight torch.Size([128, 64, 3, 3])
    # down1.maxpool_conv.1.double_conv.3.weight torch.Size([128, 128, 3, 3])

    # start to calculate mean

    data_weights = np.zeros((axis_x, axis_y))
    for idx, elem in enumerate(target_hub):
        elem_data = np.mean(np.abs(model_weights[elem].cpu().numpy()), axis=(1,2,3))
        # print(idx, elem_data)
        data_weights[idx, :len(elem_data)] = elem_data

    # plot
    plt.figure(figsize=(9,6), dpi=300)
    plt.imshow(data_weights.transpose(), cmap='hot', interpolation='nearest', aspect='auto')
    plt.xlabel("model")
    plt.ylabel("abs_weights")
    plt.title("Weights distribution over the model")
    plt.colorbar()

    plt.savefig(work_dir+"weights_{}.jpg".format(model_name))
    print(work_dir+"weights_{}.jpg".format(model_name))







