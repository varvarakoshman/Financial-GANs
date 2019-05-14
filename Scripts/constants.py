import torch
import numpy as np

learning_rate_dis = 0.01
learning_rate_gen = 0.001
learning_rate_gen_p = 0.001  # 0.01
epochs_dis = 2  # 10
epochs_gen = 2  # 20 # для сген. 30
epochs_parallel = 10  # 5
print_every = 10  # 10
plot_every = 1  # 5
m_batch_size = 4
dataset_size_to_generate = 100
hardcoded_n_in_batch = 4  # set in java code when downloading real data
size_for_basis_plot = 8
k = 1  # 1  # number of steps to apply to discriminator
weights_for_generation = [[0.7, 0.9], [0.1, 0.6]]
zero_gen_grad = torch.zeros((2, 2)).double()  # for zeroing the gradient
# weights_random = torch.from_numpy(np.array([[2., 3.], [4., 5.]]))
# weights_random = torch.from_numpy(np.array([[0.65, 0.85], [0.05, 0.55]]))
weights_random = torch.DoubleTensor(2, 2).uniform_(0, 1)
# weights_random = torch.DoubleTensor(2, 2).normal_(0, 1)
