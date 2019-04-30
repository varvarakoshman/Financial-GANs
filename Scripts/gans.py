import numpy as np
import torch
import torch.nn as nn
import csv
import time
import math
import matplotlib.pyplot as plt
from ordered_set import OrderedSet

from torch.utils.data import Dataset
from copy import deepcopy

learning_rate = 0.01
learning_rate_gen = 0.01
epochs = 5
print_every = 25
plot_every = 1
m_batch_size = 20
n_hidden = 100
dataset_size_to_generate = 1000
hardcoded_n_in_batch = 4  # set in java code when downloading real data
size_for_basis_plot = 8
weights_for_generation = [[0.7, 0.89], [0.99, 0.7]]


class CSVDataset(Dataset):
    def __init__(self):
        xs = np.loadtxt('real.csv', delimiter=',', dtype=np.float32)
        self.len = xs.shape[0]
        self.x_data = torch.from_numpy(xs[:, :])

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.len


class Generator:
    def __init__(self, indices, weights):
        self.indices = indices
        self.A = weights
        self.A.requires_grad_()

    # temp1 and temp2 - temporary matrices used to replace slicing operation
    # (which is in-place operation and not supported by autograd)
    def forward(self, x):
        x_len = len(x)
        n_layers = int(np.log2(len(x)))
        dim = tuple([2 for _ in range(n_layers)])
        x = x.view(*dim)
        result = x.double()
        for i in range(n_layers - 1, -1, -1):
            indices_to_replace = deepcopy(self.indices)
            for j in range(x_len):
                indices_to_replace[:][j][i] = [0, 1]

            indices_unique = remove_duplicates(indices_to_replace)
            for j in range(len(indices_unique)):
                temp1 = torch.ones(*dim).double()
                temp1[indices_unique[j]] = torch.tensor([0, 0]).double()
                temp2 = torch.zeros(*dim).double()
                slice = torch.mm(result[indices_unique[j]].view(*(1, 2)), torch.transpose(self.A, 0, 1)).view(2)
                temp2[indices_unique[j]] = slice
                result = result * temp1 + temp2
            # result = torch.tanh(result)  # adding non-linearity
        result = result.view(x_len, 1)
        return result

    def generate(self, x):
        out = self.forward(x)
        return out


# when converting appear duplicates of each index - remove them by making set of all of them
def remove_duplicates(indices_to_replace):
    indices_to_replace_tuple = [list(map(tuple, indices_to_replace[i])) for i in range(len(indices_to_replace))]
    indices_to_replace_set = OrderedSet(map(tuple, indices_to_replace_tuple))
    indices_to_replace_back = list(map(list, indices_to_replace_set))
    indices_to_replace = [list(map(list, indices_to_replace_back[i])) for i in
                          range(len(indices_to_replace_back))]
    return indices_to_replace


# converting indices in binary format
def prepare_indices(x):
    n_layers = int(np.log2(len(x)))
    indices_str = [np.binary_repr(i, width=n_layers) for i in range(len(x))]
    indices_list = [list(map(lambda ind: [int(ind)], index[:])) for index in indices_str]
    return indices_list


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()

        self.input_size = input_size
        # self.hidden_size = hidden_size

        self.layer1 = nn.Linear(input_size, input_size, bias=True).double()
        # self.layer2 = nn.Linear(input_size, input_size, bias=True).double()
        self.layer3 = nn.Linear(input_size, 1, bias=True).double()

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.relu(self.layer1(input))
        # x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x


def get_noise(sample_size, m_batch_size):
    input = torch.zeros(*(sample_size, m_batch_size))
    noise_sample = torch.rand_like(input)
    return noise_sample


def cross_entropy_loss(actual, label):
    if label[0] == 1:
        return -torch.log(actual)
    else:
        return -torch.log(1 - actual)


def train_gen(gen_trainable, dis, z_noise):
    gen_trainable.A.grad = torch.zeros((2, 2)).double()  # ?

    fake_all_m = torch.stack([gen_trainable.generate(z_noise[m]) for m in range(m_batch_size)]).squeeze(2)
    out = dis(fake_all_m)
    loss = cross_entropy_loss(out, torch.ones(fake_all_m.shape[0])).squeeze(1)
    loss_mean = torch.mean(loss)
    loss_mean.backward()

    gen.A.add(-learning_rate_gen, gen.A.grad)  # !

    return loss_mean


def train_dis(dis, gen, x_real, z_noise):
    # optimizer.zero_grad()
    dis.zero_grad()

    out_real = dis(x_real)
    loss_real = cross_entropy_loss(out_real, torch.ones(out_real.shape[0])).squeeze(1)
    loss_real_mean = torch.mean(loss_real)

    fake_all_m = torch.stack([gen.generate(z_noise[m]) for m in range(m_batch_size)]).squeeze(2)
    out_fake_all_m = dis(fake_all_m)
    loss_fake = cross_entropy_loss(out_fake_all_m, torch.zeros(out_fake_all_m.shape[0])).squeeze(1)
    loss_fake_mean = torch.mean(loss_fake)

    loss_mean = loss_real_mean + loss_fake_mean  # !
    loss_mean.backward()

    # optimizer = torch.optim.Adam(dis.parameters())
    # optimizer.step()

    for d_weight in dis.parameters():
        d_weight.data.add_(-learning_rate, d_weight.grad.data)
        # print("grad", d_weight.grad.data)
        # print("weights", d_weight.data)

    return loss_mean


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def plot_losses(all_losses_d, all_losses_g):  # !
    ax2 = plt.subplot()
    ax2.plot(all_losses_g, 'g', label='all_losses_g')  # !
    ax2.plot(all_losses_d, 'b', label='all_losses_d')
    ax2.legend()
    plt.show()


def plot_initial():
    noise = get_noise(1, hardcoded_n_in_batch).double()
    # noise = torch.tan(noise)
    gen = Generator(prepare_indices(noise[0]), torch.from_numpy(np.array(weights_for_generation)))
    fake = gen.generate(noise[0])

    ax = plt.subplot()
    ax.plot(fake.view(hardcoded_n_in_batch).detach().numpy(), 'g', label='generated')
    ax.plot(noise[0].numpy(), 'b', label='noise')
    ax.legend()
    plt.show()


def plot_basis(array):
    z_noise = get_noise(m_batch_size, size_for_basis_plot).double()
    gen = Generator(prepare_indices(z_noise[0]), torch.from_numpy(np.array(weights_for_generation)))

    basis = torch.from_numpy(array).double()
    fake = gen.generate(basis)

    ax = plt.subplot()
    ax.plot(fake.view(size_for_basis_plot).detach().numpy(), 'g', label='generated')
    ax.plot(basis.numpy(), 'b', label='noise')
    ax.legend()
    plt.show()


def generate_dataset():
    with open('data/generated.txt', 'w') as f:
        z_noise = get_noise(100, 4).double()
        gen = Generator(prepare_indices(z_noise[0]), torch.from_numpy(np.array(weights_for_generation)))
        for i in range(1):
            z_noise = get_noise(dataset_size_to_generate, 4).double()
            for m in range(dataset_size_to_generate):
                generated = gen.generate(z_noise[m]).detach().numpy()
                for i in range(len(generated)):
                    f.write("%f," % generated[i])
                f.write("\n")


if __name__ == '__main__':
    generate_dataset()
    # PREPARING DATA
    with open('data/generated.txt', 'r') as in_file:
        lines = in_file.read().splitlines()
        stripped = [line.replace(",", " ").split() for line in lines]
        with open('real.csv', 'w', newline='') as out_file:
            writer = csv.writer(out_file, delimiter=',', dialect='excel')
            writer.writerows(stripped)

    dataset = CSVDataset()
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=m_batch_size, shuffle=True, num_workers=1)

    sample_size = len(next(enumerate(loader))[1][0])
    set_mean = torch.mean(torch.stack([data for _, data in enumerate(loader)]))  # compute mean value for normalization

    plot_initial()
    plot_basis(np.array([0, 0, 0, 0, 0, 0, 0, 1]))
    plot_basis(np.array([0, 0, 1, 0, 0, 0, 0, 0]))
    plot_basis(np.array([0, 0, 0, 0, 0, 1, 0, 0]))
    plot_basis(np.array([1, 0, 0, 0, 0, 0, 0, 0]))

    dataset_size = dataset.len / m_batch_size
    start = time.time()
    # index, data = next(enumerate(loader))

    all_losses_g = []
    all_losses_d = []
    current_loss_g = 0
    current_loss_d = 0
    z_noise = get_noise(m_batch_size, sample_size).double()  # 1 sample for indices
    dis = Discriminator(sample_size, n_hidden)
    gen = Generator(prepare_indices(z_noise[0]), torch.from_numpy(np.array(weights_for_generation)))
    for iter in range(epochs):
        for index, data in enumerate(loader):
            z_noise = get_noise(m_batch_size, sample_size).double()
            x_real = data.double()
            x_real = torch.add(-set_mean, torch.log(x_real))  # normalization

            loss_d = train_dis(dis, gen, x_real, z_noise)
            current_loss_d += loss_d

            if index % 10 == 0:
                print('%s (%d %d%%) %.10f' % (
                    time_since(start), iter, index / dataset_size * epochs, loss_d))  # ! ,loss_g

            if index % plot_every == 0 and index != 0:
                all_losses_d.append(current_loss_d / plot_every)
                current_loss_d = 0
    print("----------")

    gen_trainable = Generator(prepare_indices(z_noise[0]), torch.rand((2, 2)).double())
    for iter in range(epochs):
        for index, data in enumerate(loader):
            z_noise = get_noise(m_batch_size, sample_size).double()
            x_real = data.double()
            x_real = torch.add(-set_mean, torch.log(x_real))  # normalization

            loss_g = train_gen(gen_trainable, dis, z_noise)  # !
            current_loss_g += loss_g  # !

            if index % 10 == 0:
                print('%s (%d %d%%) %.10f' % (
                    time_since(start), iter, index / dataset_size * 100, loss_g))  # ! ,loss_g

            if index % plot_every == 0 and index != 0:
                all_losses_g.append(current_loss_g / plot_every)
                current_loss_g = 0

    plot_losses(all_losses_d, all_losses_g)  # !
