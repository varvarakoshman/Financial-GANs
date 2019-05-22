import torch
import numpy as np

learning_rate_dis = 0.01
learning_rate_gen = 0.03
learning_rate_gen_p = 0.03  # 0.001
epochs_dis = 20  # 2
epochs_gen = 50  # 2
epochs_parallel = 30  # 10
print_every = 10  # 10
plot_every = 1  # 10
m_batch_size = 8
dataset_size_to_generate = 128
hardcoded_n_in_batch = 4  # set in java code when downloading real data
size_for_basis_plot = 8
k = 5  # 1  # number of steps to apply to generator
# weights_for_generation = [[0.9, 0.6], [0.9, 0.2]]
weights_for_generation = [[0.7, 0.9], [0.1, 0.6]]
# weights_random = torch.from_numpy(np.array([[0.65, 0.85], [0.05, 0.55]]))
# weights_random = torch.Tensor(2, 2).uniform_(0, 1)
# weights_random = torch.DoubleTensor(2, 2).normal_(0, 1)
