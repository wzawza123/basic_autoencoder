# this file is used to visualize the result

import torch
import matplotlib.pyplot as plt
import numpy as np


def show_amplitude():
    # load the amplitude npy
    base_amplitude = torch.from_numpy(np.load("./input_amplitude_0.npy"))
    loss=[]
    for i in range(0, 20):
        # load the amplitude npy
        amplitude = torch.from_numpy(np.load("./input_amplitude_{}.npy".format(i)))
        # get the loss
        loss.append(torch.sum(torch.abs(amplitude - base_amplitude))/amplitude.shape[0]/amplitude.shape[1]/amplitude.shape[2])
    plt.plot(loss)
    plt.show()

def show_phase():
    # load the amplitude npy
    base_phase = torch.from_numpy(np.load("./input_phase_0.npy"))
    loss=[]
    for i in range(0, 20):
        # load the amplitude npy
        phase = torch.from_numpy(np.load("./input_phase_{}.npy".format(i)))
        # get the loss
        loss.append(torch.sum(torch.abs(phase - base_phase))/phase.shape[0]/phase.shape[1]/phase.shape[2])
    plt.plot(loss)
    plt.show()

if __name__ == '__main__':
    # load the base torch
    show_amplitude()
    # show_phase()
