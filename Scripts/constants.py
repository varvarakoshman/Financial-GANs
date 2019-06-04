import torch
import numpy as np

learning_rate_dis = 0.001
learning_rate_gen = 0.006
learning_rate_gen_p = 0.003  # 0.001
epochs_dis = 20  # 2
epochs_gen = 50  # 2
epochs_parallel = 10   # 50
print_every = 10  # 12
plot_every = 1  # 10
m_batch_size = 8
dataset_size_to_generate = 128
hardcoded_n_in_batch = 8  # set in java code when downloading real data
size_for_basis_plot = 8
epsilon = 0.03
k = 3  # 1  # number of steps to apply to generator
weights_for_generation = [[[0.7, 0.9], [0.1, 0.6]],[[0.2, 0.3], [0.8, 0.4]]]