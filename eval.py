import pickle
import numpy as np
from matplotlib import pyplot as plt
import torchvision

def plot_loss():
    with open("./eval/loss_history.pickle", "rb") as f:
        loss = pickle.load(f)
    plt.plot(loss[:, 0], 'b', label='Training Loss')
    plt.plot(loss[:, 1], 'r', label='Validation Loss')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.show()

def show_images():
    with open("./eval/generated.pickle", "rb") as f:
        images = pickle.load(f)
    for i in range(4, 40):
        img = images[i]
        img = img.view(img.size(0), 1, 28, 28)
        grid = torchvision.utils.make_grid(img[-32:], normalize=True)
        plt.subplot(6, 6, i + 1 - 4)
        plt.imshow(grid.permute(1, 2, 0), cmap='gray')
        plt.axis('off')
    plt.show()

# plot_loss()
show_images()
