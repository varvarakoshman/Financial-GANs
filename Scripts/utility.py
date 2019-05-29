import math
import time
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch

from Scripts.constants import size_for_basis_plot, hardcoded_n_in_batch


def copy_to_csv(copy_from, copy_to):
    with open(copy_from, 'r') as in_file:
        lines = in_file.read().splitlines()
        stripped = [line.replace(",", " ").split() for line in lines]
        with open(copy_to, 'w', newline='') as out_file:
            writer = csv.writer(out_file, delimiter=',', dialect='excel')
            writer.writerows(stripped)


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def get_noise(sample_size, m_batch_size):
    input = torch.zeros(*(sample_size, m_batch_size))
    noise_sample = torch.rand_like(input)
    # noise_sample = torch.DoubleTensor(sample_size, m_batch_size).normal_(0, 1)
    return noise_sample


def plot_losses(all_losses_d, all_losses_g):
    ax1 = plt.subplot()
    ax1.plot(all_losses_d, 'b', label='all_losses_d')
    ax1.legend()
    plt.show()
    ax2 = plt.subplot()
    ax2.plot(all_losses_g, 'g', label='all_losses_g')
    ax2.legend()
    plt.show()


def plot_losses_together(all_losses_d, all_losses_g):
    ax1 = plt.subplot()
    ax1.plot(all_losses_d, 'b', label='all_losses_d')
    ax1.plot(all_losses_g, 'g', label='all_losses_g')
    ax1.legend()
    plt.show()


def plot_dis_accuracy(dis_accuracy):
    ax1 = plt.subplot()
    ax1.plot(dis_accuracy, 'o', label='dis_accuracy')
    ax1.legend()
    plt.show()


def plot_gradient(gradient):
    # print(gradient)
    ax1 = plt.subplot()
    ax1.plot(gradient, 'r', label='gradient')
    ax1.legend()
    plt.show()


def plot_initial(gen, noise):
    fake = gen.generate(noise[0])
    ax = plt.subplot()
    ax.plot(fake.view(hardcoded_n_in_batch).detach().numpy(), 'g', label='generated')
    ax.plot(noise[0].numpy(), 'b', label='noise')
    ax.legend()
    plt.show()


def plot_basis(gen):
    basis_coordinates = [np.array([0, 0, 0, 0, 0, 0, 0, 1]), np.array([1, 0, 0, 0, 0, 0, 0, 0]),
                         np.array([0, 0, 1, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 1, 0, 0])]
    for array in basis_coordinates:
        basis = torch.from_numpy(array).double()
        fake = gen.generate(basis)
        # print(fake.view(size_for_basis_plot).detach().numpy())
        ax = plt.subplot()
        ax.plot(fake.view(size_for_basis_plot).detach().numpy(), 'g', label='generated')
        ax.plot(basis.numpy(), 'b', label='noise')
        ax.legend()
        plt.show()


def plot_gen_true_fake(gen_trained, gen_clever, sample_size, noise):
    for i in range(1):
        ax1 = plt.subplot()
        ax1.plot(gen_trained.generate(noise[0]).detach().numpy(), 'g', label='generated_fake')
        ax1.legend()
        plt.show()
        ax2 = plt.subplot()
        ax2.plot(gen_clever.generate(noise[0]).detach().numpy(), 'b', label='true')
        ax2.legend()
        plt.show()
