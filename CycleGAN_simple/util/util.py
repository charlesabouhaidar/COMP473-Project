"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def loss_to_graph(experiment_name, path_to_loss_file):
    counter = 1
    G_A_Loss = []
    G_B_Loss = []
    D_A_Loss = []
    D_B_Loss = []
    Cycle_A = []
    Cycle_B = []
    idt_A = []
    idt_B = []

    #line looks likes
    #(epoch: 9, iters: 2891, time: 0.371, data: 0.001) D_A: 0.123 G_A: 0.724 cycle_A: 1.047 idt_A: 0.401 D_B: 0.145 G_B: 0.715 cycle_B: 0.866 idt_B: 0.478
    with open(path_to_loss_file) as fp:
        for i, line in enumerate(fp):
            if not (line[0] ==  "="): #skip first line of stats and any line that doesn't have any stats
	            position = line.find("G_A: ")
	            G_A_Loss.append(line[position+5 : position +10])

	            position = line.find("G_B: ")
	            G_B_Loss.append(line[position+5 : position +10])

	            position = line.find("D_A: ")
	            D_A_Loss.append(line[position+5 : position +10])

	            position = line.find("D_B: ")
	            D_B_Loss.append(line[position+5 : position +11])

	            position = line.find("cycle_A: ")
	            Cycle_A.append(line[position+9 : position +15])

	            position = line.find("cycle_B: ")
	            Cycle_B.append(line[position+9 : position +15])

	            position = line.find("idt_A: ")
	            idt_A.append(line[position+7 : position +13])

	            position = line.find("idt_B: ")
	            idt_B.append(line[position+7 : position +13])

	            counter = counter +1

    #Generator losses
    G_A = list(map(float, G_A_Loss))
    G_B = list(map(float, G_B_Loss))
    # Discriminator losses
    D_A = list(map(float, D_A_Loss))
    D_B = list(map(float, D_B_Loss))
    # Cycle consistency losses
    C_A = list(map(float, Cycle_A))
    C_B = list(map(float, Cycle_B))
    # Identity losses
    ID_A = list(map(float, idt_A))
    ID_B = list(map(float, idt_B))
    
    #x axis
    x = np.arange(len(G_A))

    #y axis
    fig = plt.figure(figsize=(16,8))
    fig.suptitle(experiment_name)
    ax1 = fig.add_subplot(221) ## generator plot
    ax2 = fig.add_subplot(222) ## discriminator plot
    ax3 = fig.add_subplot(223) ## Cycle consistency plot
    ax4 = fig.add_subplot(224) ## Identity loss plot

    #Graph 1
    ax1.set_title('Generator Losses')
    ax1.set_ylabel('error loss')
    ax1.plot(x,G_A, color="seagreen", label="Generator A")
    ax1.plot(x,G_B, color="orchid", label="Generator B")
    ax1.legend()
    #Graph 2
    ax2.set_title('Discriminator Losses')
    ax2.plot(x,D_A, color="seagreen", label="Discriminator A")
    ax2.plot(x,D_B, color="orchid", label="Discriminator B")
    ax2.legend()
    #Graph 3
    ax3.set_title('Cycle losses')
    ax3.set_xlabel('epochs')
    ax3.set_ylabel('error loss')
    ax3.plot(x,C_A, color="seagreen", label="Cycle A")
    ax3.plot(x,C_B, color="orchid", label="Cycle B")
    ax3.legend()
    #Graph 4
    ax4.set_title('Identity losses')
    ax4.set_xlabel('epochs')
    ax4.plot(x,ID_A, color="seagreen", label="Forward Identity")
    ax4.plot(x,ID_B, color="orchid", label="Backward Identity")
    ax4.legend()