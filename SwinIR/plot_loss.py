import numpy as np
import matplotlib.pyplot as plt
import glob
import copy

n_epoch = 50
folder = "./bridge_small/*/"

model_hub = []
npy_list = sorted(glob.glob(folder+"loss/epoch_loss_*.npy"))
for npy_path in npy_list:
    # print(npy_path)
    path_split = npy_path.split("/")
    model_name = path_split[2]+"_"+path_split[4][-9]
    # print(model_name)
    if not model_name in model_hub:
        model_hub.append(model_name)
print(model_hub)

loss = np.zeros((n_epoch))
plot_target = []
for model_name in model_hub:
    current_package = [model_name]
    for idx in range(n_epoch):
        num = "{:03d}".format(idx+1)
        name = "./bridge_small/{}/loss/epoch_loss_{}_{}.npy".format(model_name[:-2], model_name[-1], num)
        data = np.load(name)
        loss[idx] = np.mean(data)
    current_package.append(copy.deepcopy(loss))
    plot_target.append(current_package)


legend_list = []
plt.figure(figsize=(9,6), dpi=300)
for package in plot_target:
    loss_array = package[1]
    loss_tag = package[0]
    legend_list.append(loss_tag)
    print(loss_tag, np.mean(loss_array))
    plt.plot(range(n_epoch), loss_array)

plt.xlabel("epoch")
plt.ylabel("loss")
plt.yscale("log")
plt.legend(["training", "validation"])
plt.title("Training curve")

plt.savefig("./bridge_small/loss_{}.jpg".format(n_epoch))