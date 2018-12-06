import matplotlib.pyplot as plt
import torch

def plot_output(tile, output, target):

    b,c,h,w = output.shape
    for idx in range(b):

        o = torch.argmax(output[idx],0).cpu()
        t = target[idx]

        plt.figure()
        plt.imshow(o)
        plt.suptitle("output", y=1.05, fontsize=18)
        plt.colorbar()
        plt.title(tile[idx], fontsize=10)

        plt.figure()
        plt.imshow(t)
        plt.suptitle("output", y=1.05, fontsize=18)
        plt.colorbar()
        plt.title(tile[idx], fontsize=10)

    plt.show()