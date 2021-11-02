import numpy as np
import matplotlib.pyplot as plt
import glob

folder = "./xue_5to1/"

npy_list = sorted(glob.glob(folder+"npy/epoch_loss_*.npy"))
for npy_path in npy_list:
    print(npy_path)

n_epoch = 900
loss_t = np.zeros((n_epoch))
loss_v = np.zeros((n_epoch))

for idx in range(n_epoch):
    num = "{:03d}".format(idx+1)
    name_t = folder+"npy/epoch_loss_t_{}.npy".format(num)
    name_v = folder+"npy/epoch_loss_v_{}.npy".format(num)
    data_t = np.load(name_t)
    data_v = np.load(name_v)
    loss_t[idx] = np.mean(data_t)
    loss_v[idx] = np.mean(data_v)

plt.figure(figsize=(9,6), dpi=300)
plt.plot(range(n_epoch), loss_t)
plt.plot(range(n_epoch), loss_v)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.yscale("log")
plt.legend(["training", "validation"])
plt.title("Training curve")

plt.savefig(folder+"loss_5to1.jpg")