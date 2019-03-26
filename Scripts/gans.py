import numpy as np
import torch
import torch.nn as nn
import csv
import time
import math
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from copy import deepcopy

learning_rate = 0.01
epochs = 100
k = 1  # number of steps to apply to the discriminator
print_every = 10
plot_every = 5
m_batch_size = 11
n_hidden = 100


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
    def __init__(self, indices):
        self.indices = indices
        self.A = torch.from_numpy(np.array(
            [[-2.0772613300000002, 0.60428355], [1.39508315, 0.60428355]]))  # TODO: make A randomly initialized
        self.A.requires_grad_()

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

            indices_to_replace_unique = remove_duplicates(indices_to_replace)
            for j in range(len(indices_to_replace_unique)):
                result[indices_to_replace_unique[j]] = torch.mm(result[indices_to_replace_unique[j]].unsqueeze(0),
                                                                torch.transpose(self.A, 0, 1))
            result = torch.tanh(result)
        return result

    def generate(self, x):
        out = self.forward(x)
        return out


def remove_duplicates(indices_to_replace):
    indices_to_replace_tuple = [list(map(tuple, indices_to_replace[i])) for i in range(len(indices_to_replace))]
    indices_to_replace_set = set(map(tuple, indices_to_replace_tuple))
    indices_to_replace_back = list(map(list, indices_to_replace_set))
    indices_to_replace = [list(map(list, indices_to_replace_back[i])) for i in
                          range(len(indices_to_replace_back))]
    return indices_to_replace


def prepare_indices(x):
    n_layers = int(np.log2(len(x)))
    indices_str = [np.binary_repr(i, width=n_layers) for i in range(len(x))]
    indices_list = [list(map(lambda ind: [int(ind)], index[:])) for index in indices_str]
    return indices_list


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.layer1 = nn.Linear(input_size, hidden_size).double()
        self.layer2 = nn.Linear(hidden_size, hidden_size).double()
        self.layer3 = nn.Linear(hidden_size, hidden_size).double()
        self.layer4 = nn.Linear(hidden_size, hidden_size).double()
        self.layer5 = nn.Linear(hidden_size, hidden_size).double()
        self.layer6 = nn.Linear(hidden_size, hidden_size).double()
        self.layer7 = nn.Linear(hidden_size, 1).double()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.relu(self.layer1(input))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.relu(self.layer6(x))
        x = self.sigmoid(self.layer7(x))
        return x


def noise(sample_size, m_batch_size):
    noise_sample = torch.randn(sample_size, m_batch_size)
    return noise_sample


def cross_entropy_loss(actual, label):
    if label[0] == 1:
        return -torch.log(actual)
    else:
        return -torch.log(1 - actual)


# TODO: update in a batch, correct dimensions

def train_gen(gen, dis, z_noise):
    fake_all_m = torch.stack([gen.generate(z_noise[m]) for m in range(m_batch_size)])
    out = dis(fake_all_m)
    loss = cross_entropy_loss(out, torch.ones(fake_all_m.shape[0]))

    loss.backward()

    gen.A.add_(-learning_rate, gen.A.grad.data)
    return loss


# TODO: run the same with optimizer
def train_dis(dis, gen, x_real, z_noise):
    out_real = dis.forward(x_real)
    loss_real = cross_entropy_loss(out_real, torch.ones(out_real.shape[0]))

    fake_all_m = torch.stack([gen.generate(z_noise[m]) for m in range(m_batch_size)])
    print(fake_all_m)
    out_fake_all_m = dis(fake_all_m)
    loss_fake = cross_entropy_loss(out_fake_all_m, torch.zeros(out_fake_all_m.shape[0]))

    loss = loss_real + loss_fake
    loss.backward()

    d_weight = dis.parameters()[0]
    d_weight.data.add_(-learning_rate, d_weight.grad.data)

    return loss


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


if __name__ == "__main__":
    # PREPARING DATA
    # TODO: take not real data, but white noise
    with open('result.txt', 'r') as in_file:
        lines = in_file.read().splitlines()
        stripped = [line.replace(",", " ").split() for line in lines]
        with open('real.csv', 'w', newline='') as out_file:
            writer = csv.writer(out_file, delimiter=',', dialect='excel')
            writer.writerows(stripped)

    dataset = CSVDataset()
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=m_batch_size, shuffle=True, num_workers=1)

    '''
    for batch_idx, data in enumerate(loader):
        print('batch: {}\tdata: {}'.format(batch_idx, data))  # print dataset
    '''
    sample_size = len(next(enumerate(loader))[1][0])
    set_mean = torch.mean(torch.stack([data for _, data in enumerate(loader)]))  # compute mean value for normalization

    all_losses_g = []
    all_losses_d = []
    current_loss_g = 0
    current_loss_d = 0

    start = time.time()
    for epoch in range(1, epochs + 1):
        for k_step in range(k):
            for _, (data) in enumerate(loader):
                print(data)
                z_noise = noise(m_batch_size, sample_size).double()
                x_real = data.double()  # TODO: apply normalization

                gen = Generator(prepare_indices(z_noise[0]))
                dis = Discriminator(sample_size, n_hidden)

                loss_d = train_dis(dis, gen, x_real, z_noise)
                loss_g = train_gen(gen, dis, z_noise)
                current_loss_d += loss_d
                current_loss_g += loss_g

                if epoch % print_every == 0:
                    print('%s (%d %d%%) %.10f %10f' % (time_since(start), epoch, epoch / epochs * 100, loss_d, loss_g))

                if epoch % plot_every == 0:
                    all_losses_d.append(current_loss_d / plot_every)
                    all_losses_g.append(current_loss_g / plot_every)
                    current_loss_d = 0
                    current_loss_g = 0
    plt.figure()
    plt.plot(all_losses_g)
    plt.plot(all_losses_d)
