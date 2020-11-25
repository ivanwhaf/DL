# @Author: Ivan
# @Time: 2020/9/11
# This file contains GAN model

import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt


epochs = 100
batch_size = 64
lr = 0.0003  # better be low!


def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),  # convert (0,255) to (0,1)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # transforms.Normalize((0.5,), (0.5,))  # convert (0,1) to (-1,1)
    ])

    # train_dataset = datasets.MNIST(
    #     './dataset', train=True, download=True, transform=transform)
    train_dataset = datasets.CIFAR10(
        './dataset', train=True, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 3, 32, 32)
    return out


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.generator(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.discriminator(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    train_loader = load_dataset()
    # generator = Generator(100, 256, 784).to(device)
    # discriminator = Discriminator(784, 256, 1).to(device)
    nosie_size = 256
    generator = Generator(nosie_size, 1024, 3072).to(device)
    discriminator = Discriminator(3072, 1024, 1).to(device)

    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    criterion = nn.BCELoss().to(device)
    for epoch in range(epochs):
        for idx, (inputs, _) in enumerate(train_loader):
            num_img = inputs.size(0)
            # flatten imgs
            inputs = inputs.view(num_img, -1)
            # convert tensor to variable
            real_img = Variable(inputs).to(device)

            # define label equal 1
            real_label = Variable(torch.ones(num_img)).cuda()
            # define label equal 0
            fake_label = Variable(torch.zeros(num_img)).cuda()

            # =================train discriminator===================
            # calculate real loss
            real_out = discriminator(real_img)
            loss_real = criterion(real_out, real_label)

            # calculate fake loss
            noise = Variable(torch.randn(num_img, nosie_size)).cuda()
            fake_img = generator(noise).detach()
            fake_out = discriminator(fake_img)
            loss_fake = criterion(fake_out, fake_label)

            # back prop
            d_loss = loss_real+loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ==================train generator=====================
            noise = Variable(torch.randn(num_img, nosie_size)).cuda()
            fake_img = generator(noise)
            d_out = discriminator(fake_img)

            g_loss = criterion(d_out, real_label)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if (idx + 1) % 100 == 0:
                print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f},D real: {:.6f},D fake: {:.6f}'.format(
                    epoch, epochs, d_loss.data.item(), g_loss.data.item(),
                    real_out.data.mean(), fake_out.data.mean()))

        fake_images = to_img(fake_img.cpu().data)
        save_image(fake_images, './img/cifar10/epoch-{}.png'.format(epoch))

    # save model
    torch.save(generator, "./model/gan/cifar10/generator.pth")
    torch.save(discriminator, "./model/gan/cifar10/discriminator.pth")
