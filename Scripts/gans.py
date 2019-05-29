import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from ordered_set import OrderedSet
from torch.utils.data import Dataset

from Scripts.constants import learning_rate_gen, m_batch_size, weights_for_generation, dataset_size_to_generate, \
    epochs_parallel, k, plot_every, hardcoded_n_in_batch, size_for_basis_plot, epochs_gen, epochs_dis, \
    learning_rate_gen_p, print_every, learning_rate_dis, epsilon
from Scripts.utility import plot_gradient, time_since, plot_losses, get_noise, copy_to_csv, plot_dis_accuracy, \
    plot_losses_together, plot_gen_true_fake

feature_matching_loss = nn.MSELoss()


class CSVDataset(Dataset):
    def __init__(self, name):
        xs = np.loadtxt(name, delimiter=',', dtype=np.float32)
        self.len = xs.shape[0]
        self.x_data = torch.from_numpy(xs[:, :])

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.len


class Generator:
    def __init__(self, indices, weights):
        self.indices = indices
        # self.A = Variable(weights, requires_grad=True)
        self.A = weights
        self.A.requires_grad_()

    # temp1 and temp2 - temporary matrices used to replace slicing operation
    # (which is in-place operation and not supported by autograd)
    def forward(self, x):
        x_len = len(x)
        n_times = int(np.log2(len(x)))
        dim = tuple([2 for _ in range(n_times)])
        x = x.view(*dim)
        result = x.double()
        for i in range(n_times - 1, -1, -1):
            indices_to_replace = deepcopy(self.indices)
            for j in range(x_len):
                indices_to_replace[:][j][i] = [0, 1]

            indices_unique = remove_duplicates(indices_to_replace)
            for j in range(len(indices_unique)):
                temp1 = torch.ones(*dim).double()
                temp1[indices_unique[j]] = torch.tensor([0, 0]).double()
                temp2 = torch.zeros(*dim).double()
                slice = torch.mm(result[indices_unique[j]].view(*(1, 2)), torch.transpose(self.A, 0, 1).double()).view(
                    2)
                temp2[indices_unique[j]] = slice
                result = result * temp1 + temp2
        result = torch.tanh(result)  # adding non-linearity
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
    def __init__(self, input_size):
        super(Discriminator, self).__init__()

        self.input_size = input_size

        self.layer1 = nn.Linear(input_size, input_size, bias=True).double()
        self.layer2 = nn.Linear(input_size, 1, bias=True).double()

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        features = self.relu(self.layer1(input))
        x = self.sigmoid(self.layer2(features))
        # x = self.layer2(features)
        return features, x


def binary_cross_entropy(actual, label):
    if label[0] == 1:
        return -torch.log(actual)
    else:
        return -torch.log(1 - actual)


def train_gen(gen_trainable, gen_fixed_clever, dis_fixed_silly, z_noise, alpha, optimizer_gen):
    optimizer_gen.zero_grad()
    # _fake_all_m = torch.stack([gen_trainable.generate(z_noise[m]) for m in range(m_batch_size)]).squeeze(2)
    # print(_fake_all_m)
    # _, out = dis_fixed_silly(_fake_all_m)
    # real_all_m = torch.stack([gen_fixed_clever.generate(z_noise[m]) for m in range(m_batch_size)]).squeeze(2)
    # _, out2 = dis_fixed_silly(real_all_m)
    # loss = binary_cross_entropy(out, torch.ones(_fake_all_m.shape[0])).squeeze(1)
    fake_all_m = torch.stack([gen_trainable.generate(z_noise[m]) for m in range(m_batch_size)]).squeeze(2)
    real_all_m = torch.stack([gen_fixed_clever.generate(z_noise[m]) for m in range(m_batch_size)]).squeeze(2)
    out_fake_features, _ = dis_fixed_silly(fake_all_m)
    out_real_features, _ = dis_fixed_silly(real_all_m)
    loss = feature_matching_loss(out_fake_features, out_real_features)
    loss_mean = torch.mean(loss)
    loss_mean.backward()
    # real_all_m = torch.stack([gen_fixed_clever.generate(z_noise[m]) for m in range(m_batch_size)]).squeeze(2)
    # loss = feature_matching_loss(_fake_all_m, torch.zeros(_fake_all_m.shape[0], _fake_all_m.shape[1]).double())
    # loss = feature_matching_loss(_fake_all_m, real_all_m)
    # loss_mean = torch.mean(loss)
    # print(loss_mean)
    # loss_mean.backward()
    # loss.backward()
    grad_norm = torch.norm(gen_trainable.A.grad).numpy()
    optimizer_gen.step()
    # print(gen_trainable.A.grad)
    # gen_trainable.A = gen_trainable.A - alpha * gen_trainable.A.grad
    return loss_mean, grad_norm


def train_dis(d_trainable, g_fixed_silly, x_real, z_noise, optimizer):
    optimizer.zero_grad()
    fake_all_m = torch.stack([g_fixed_silly.generate(z_noise[m]) for m in range(m_batch_size)]).squeeze(2)
    _, out_real = d_trainable(x_real)  # + torch.DoubleTensor(m_batch_size, sample_size).normal_(0, 1))  # NOISE ADDED
    loss_real = binary_cross_entropy(out_real, torch.ones(out_real.shape[0]).long()).squeeze(1)
    _, out_fake_all_m = d_trainable(fake_all_m)
    loss_fake = binary_cross_entropy(out_fake_all_m, torch.zeros(out_fake_all_m.shape[0]).long()).squeeze(1)
    loss_mean = torch.mean(loss_fake) + torch.mean(loss_real)
    loss_mean.backward()
    optimizer.step()
    dis_accuracy.append(torch.mean(out_real).detach().numpy())
    dis_accuracy.append(torch.mean(out_fake_all_m).detach().numpy())
    return loss_mean


def dis_training_cycle(mode, loader):
    # optimizer = torch.optim.SGD(dis_trainable.parameters(), lr=learning_rate_dis, momentum=0.9)
    optimizer = torch.optim.Adam(dis_trainable.parameters(), lr=learning_rate_dis)
    all_losses_d = []
    current_loss_d = 0
    start = time.time()
    for iter in range(epochs_dis):
        for index, data in enumerate(loader):
            z_noise = get_noise(m_batch_size, sample_size).double()
            x_real = data.double()
            if mode == 'real':
                x_real = torch.add(-set_mean, torch.log(x_real))  # normalization
            loss_d = train_dis(dis_trainable, gen_fixed_silly, x_real, z_noise, optimizer)
            current_loss_d += loss_d
            if (index + 1) % print_every == 0:
                print('%s (%d %d%%) %.10f' % (time_since(start), iter, index / dataset_size * 100, loss_d))
            if (index + 1) % plot_every == 0:
                all_losses_d.append(current_loss_d / plot_every)
                current_loss_d = 0
    return all_losses_d


def dis_pre_training_cycle(mode, loader):
    optimizer = torch.optim.Adam(dis_silly.parameters(), lr=learning_rate_dis)
    all_losses_d = []
    current_loss_d = 0
    start = time.time()
    for iter in range(epochs_dis):
        for index, data in enumerate(loader):
            z_noise = get_noise(m_batch_size, sample_size).double()
            x_real = data.double()
            if mode == 'real':
                x_real = torch.add(-set_mean, torch.log(x_real))  # normalization
            if not (0.45 - epsilon < torch.mean(dis_silly(x_real)[1]) < 0.45 + epsilon \
                    or 0.55 - epsilon < torch.mean(dis_silly(x_real)[1]) < 0.55 + epsilon):
                loss_d = train_dis(dis_silly, gen_fixed_silly, x_real, z_noise, optimizer)
                current_loss_d += loss_d
                if (index + 1) % print_every == 0:
                    print('%s (%d %d%%) %.10f' % (time_since(start), iter, index / dataset_size * 100, loss_d))
                if (index + 1) % plot_every == 0:
                    all_losses_d.append(current_loss_d / plot_every)
                    current_loss_d = 0
            else:
                return all_losses_d
    return all_losses_d


def gen_training_cycle(loader):
    all_losses_g = []
    current_loss_g = 0
    gradients = []
    start = time.time()
    optimizer = torch.optim.Adam([gen_fixed_silly.A], lr=learning_rate_gen)
    # sheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
    # optimizer = torch.optim.SGD([gen_fixed_silly.A], lr=learning_rate_gen, momentum=0.9)
    # optimizer = torch.optim.SGD([gen_fixed_silly.A], lr=learning_rate_gen)
    # optimizer = torch.optim.ASGD([gen_fixed_silly.A], lr=learning_rate_gen)
    for iter in range(epochs_gen):
        for index, data in enumerate(loader):
            z_noise = get_noise(m_batch_size, sample_size).double()
            loss_g, grad_norm = train_gen(gen_fixed_silly, gen_fixed_clever, dis_trainable, z_noise,
                                          learning_rate_gen, optimizer)
            gradients.append(grad_norm)
            current_loss_g += loss_g
            if (index + 1) % print_every == 0:
                print('%s (%d %d%%) %.10f' % (
                    time_since(start), iter, index / dataset_size * 100, loss_g))
            if (index + 1) % plot_every == 0:
                # if index % 1 == 0:
                all_losses_g.append(current_loss_g / plot_every)
                current_loss_g = 0
    return all_losses_g, gradients


def train_parallel_cycle(mode, loader):
    all_losses_g = []
    all_losses_d = []
    gradients = []
    current_loss_g = 0
    current_loss_d = 0
    # optimizer = torch.optim.SGD(dis_silly.parameters(), lr=0.005, momentum=0.9)
    # optimizer = torch.optim.SGD(dis_silly.parameters(), lr=learning_rate_dis, momentum=0.9)
    optimizer = torch.optim.Adam(dis_silly.parameters(), lr=learning_rate_dis)
    # optimizer_gen = torch.optim.SGD([gen_fixed_silly.A], lr=learning_rate_gen_p, momentum=0.9)
    optimizer_gen = torch.optim.Adam([gen_silly.A], lr=learning_rate_gen_p)
    # optimizer = torch.optim.SGD(dis_silly.parameters(), lr=0.001)
    start = time.time()
    for epoch in range(epochs_parallel):
        for index, data in enumerate(loader):
            for k_step in range(k):
                z_noise = get_noise(m_batch_size, sample_size).double()
                x_real = data.double()
                if mode == 'real':
                    x_real = torch.add(-set_mean, torch.log(x_real))  # normalization
                loss_d = train_dis(dis_silly, gen_silly, x_real, z_noise, optimizer)
                current_loss_d += loss_d
            z_noise = get_noise(m_batch_size, sample_size).double()
            loss_g, grad = train_gen(gen_silly, gen_fixed_clever, dis_silly, z_noise, learning_rate_gen_p,
                                     optimizer_gen)
            current_loss_g += loss_g
            gradients.append(grad)
            if (index + 1) % print_every == 0:  # 1
                print('%s (%d %d%%) %.10f %10f' % (
                    time_since(start), epoch, (index + 1) / dataset_size * 100, loss_d, loss_g))
            if (index + 1) % plot_every == 0:
                all_losses_d.append(current_loss_d / plot_every)
                all_losses_g.append(current_loss_g / plot_every)
                current_loss_d = 0
                current_loss_g = 0
    return all_losses_d, all_losses_g, gradients


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


def plot_all():
    noise_for_initial_plot = get_noise(1, hardcoded_n_in_batch).double()
    gen_initial = Generator(prepare_indices(noise_for_initial_plot[0]),
                            torch.from_numpy(np.array(weights_for_generation)))
    # plot_initial(gen_initial, noise_for_initial_plot)

    z_noise_basis = get_noise(m_batch_size, size_for_basis_plot).double()
    gen_basis = Generator(prepare_indices(z_noise_basis[0]), torch.from_numpy(np.array(weights_for_generation)))
    # plot_basis(gen_basis)

    plot_losses_together(losses_d_parallel, losses_g_parallel)
    # plot_gradient(grad_both)
    plot_dis_accuracy(dis_accuracy)
    # plot_gradient(grad_gen)
    # plot_losses(losses_d, losses_g)


def test_parallel():
    for i in range(10):
        noise = get_noise(m_batch_size, sample_size).double()
        real_for_pred = torch.stack([gen_fixed_clever.generate(noise[m]) for m in range(m_batch_size)]).squeeze(2)
        fake_for_pred = torch.stack([gen_silly.generate(noise[m]) for m in range(m_batch_size)]).squeeze(2)
        _, prediction_fake = dis_silly(fake_for_pred)
        _, prediction_real = dis_silly(real_for_pred)
        print("chance that fake is taken for real: ", torch.mean(prediction_fake))
        print("chance that real is taken for real: ", torch.mean(prediction_real))


def test_parallel_real():
    copy_to_csv('data/real_test.txt', 'test.csv')
    dataset_test = CSVDataset('test.csv')
    loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=m_batch_size, shuffle=True,
                                              num_workers=1)
    set_mean_test = torch.mean(
        torch.stack([data for _, data in enumerate(loader_test)]))  # compute mean value for normalization
    for index, data in enumerate(loader_test):
        fake_for_pred = torch.stack([gen_silly.generate(noise[m]) for m in range(m_batch_size)]).squeeze(2)
        x_real = data.double()
        x_real = torch.add(-set_mean_test, torch.log(x_real))  # normalization
        _, prediction_fake = dis_silly(fake_for_pred)
        _, prediction_real = dis_silly(x_real)
        print("chance that fake is taken for real: ", prediction_fake)
        print("chance that real is taken for real: ", prediction_real)


if __name__ == '__main__':
    dis_accuracy = []
    # loading generated dataset
    generate_dataset()
    copy_to_csv('data/generated.txt', 'generated_copy.csv')
    dataset_gen = CSVDataset('generated_copy.csv')
    loader = torch.utils.data.DataLoader(dataset=dataset_gen, batch_size=m_batch_size, shuffle=True, num_workers=1)
    sample_size = len(next(enumerate(loader))[1][0])
    dataset_size = dataset_gen.len / m_batch_size
    noise = get_noise(m_batch_size, sample_size).double()  # 1 sample for indices

    # loading real dataset
    '''copy_to_csv('data/result4.txt', 'real.csv')
    dataset_real = CSVDataset('real.csv')
    loader_real = torch.utils.data.DataLoader(dataset=dataset_real, batch_size=m_batch_size, shuffle=True,
                                              num_workers=1)
    dataset_size_real = dataset_real.len / m_batch_size
    set_mean = torch.mean(
        torch.stack([data for _, data in enumerate(loader_real)]))  # compute mean value for normalization
    sample_size = len(next(enumerate(loader_real))[1][0])
    dataset_size = dataset_real.len / m_batch_size
    noise = get_noise(m_batch_size, sample_size).double()  # 1 sample for'''

    # 1. train discriminator
    set_mean = 0
    gen_fixed_clever = Generator(prepare_indices(noise[0]), torch.from_numpy(np.array(weights_for_generation)))
    dis_trainable = Discriminator(sample_size)
    # dis_trainable.apply(weights_init)
    # dis_trainable.layer1.weight.data.fill_(0)
    # dis_trainable.layer1.bias.data.fill_(0)
    # dis_trainable.layer2.weight.data.fill_(0)
    # dis_trainable.layer2.bias.data.fill_(0.45)

    weights_random = torch.Tensor(2, 2).uniform_(0, 1)
    # weights_random = torch.from_numpy(np.array([[0.65, 0.85], [0.05, 0.55]]))
    gen_fixed_silly = Generator(prepare_indices(noise[0]), weights_random)
    losses_d = []
    # losses_d = dis_training_cycle('generated', loader)
    # losses_d = dis_training_cycle('real', loader_real)

    # real = torch.stack([gen_fixed_clever.generate(noise[m]) for m in range(m_batch_size)]).squeeze(2)
    # print("chance of real data to be taken as real: ", dis_trainable(real)[1])
    # fake = torch.stack([gen_fixed_silly.generate(noise[m]) for m in range(m_batch_size)]).squeeze(2)
    # print("chance of fake data to be taken as real: ", dis_trainable(fake)[1])
    # plot_losses(losses_d, [])

    # 2. train generator
    dis_silly_for_gen = Discriminator(sample_size)
    losses_g = []
    print("random generator's weights: ", gen_fixed_silly.A)
    plot_gen_true_fake(gen_fixed_silly, gen_fixed_clever, sample_size, noise)
    losses_g, grad_gen = gen_training_cycle(loader)
    z_noise = get_noise(m_batch_size, sample_size).double()
    fake_all_m = torch.stack([gen_fixed_silly.generate(z_noise[m]) for m in range(m_batch_size)]).squeeze(2)
    _, out_real = dis_silly_for_gen(fake_all_m)
    print(out_real)
    plot_gen_true_fake(gen_fixed_silly, gen_fixed_clever, sample_size, noise)
    # losses_g, grad_gen = gen_training_cycle(loader_real)
    print("trained generator's weights: ", gen_fixed_silly.A)
    plot_losses(losses_d, losses_g)
    plot_gradient(grad_gen)
    # 3. training in parallel
    dis_accuracy = []
    dis_silly = Discriminator(sample_size)
    # real = torch.stack([gen_fixed_clever.generate(noise[m]) for m in range(m_batch_size)]).squeeze(2)
    # print("chance of real data to be taken as real: ", dis_silly(real)[1])
    # fake = torch.stack([gen_fixed_silly.generate(noise[m]) for m in range(m_batch_size)]).squeeze(2)
    # print("chance of fake data to be taken as real: ", dis_silly(fake)[1])

    # losses_d_pre = dis_pre_training_cycle('generated', loader)

    real = torch.stack([gen_fixed_clever.generate(noise[m]) for m in range(m_batch_size)]).squeeze(2)
    fake = torch.stack([gen_fixed_silly.generate(noise[m]) for m in range(m_batch_size)]).squeeze(2)
    print("chance of real data to be taken as real: ", dis_silly(real)[1])
    print("chance of fake data to be taken as real: ", dis_silly(fake)[1])
    # plot_losses(losses_d, [])

    gen_silly = Generator(prepare_indices(noise[0]), torch.Tensor(2, 2).uniform_(0, 1))
    print("random weights for gen: ", gen_silly.A)
    noise_2 = get_noise(1, sample_size).double()
    plot_gen_true_fake(gen_silly, gen_fixed_clever, sample_size, noise_2)
    losses_d_parallel, losses_g_parallel, grad_both = train_parallel_cycle('generated', loader)
    real_after = torch.stack([gen_fixed_clever.generate(noise[m]) for m in range(m_batch_size)]).squeeze(2)
    fake_after = torch.stack([gen_silly.generate(noise[m]) for m in range(m_batch_size)]).squeeze(2)
    print("chance of real data to be taken as real: ", torch.mean(dis_silly(real_after)[1]))
    print("chance of fake data to be taken as real: ", torch.mean(dis_silly(fake_after)[1]))
    plot_gen_true_fake(gen_silly, gen_fixed_clever, sample_size, noise_2)
    # losses_d_parallel, losses_g_parallel, grad_both = train_parallel_cycle('real', loader_real)
    print("trained generator's weights: ", gen_silly.A)
    test_parallel()

    x_real = next(enumerate(loader))[1]

    # print(dis_trainable(x_real.double()))
    # exit(1)

    # plot_all()
    plot_losses_together(losses_d_parallel, losses_g_parallel)
    plot_gradient(grad_both)
    # test_parallel_real()
