# @Author: Ivan
# @Time: 2020/9/8
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
from models import LeNet, AlexNet, VGGNet, GoogLeNet, ResNet, CIFAR10Net, CIFAR10LeNet

batch_size = 64
epochs = 150


def load_dataset():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(
        './dataset', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(
        './dataset', train=False, transform=transform, download=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(model, train_loader, optimizer, epoch, device, train_loss_lst, train_acc_lst):
    model.train()  # Sets the module in training mode
    correct = 0
    train_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        pred = outputs.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()

        # criterion = nn.CrossEntropyLoss()
        # loss = criterion(inputs, labels)
        loss = F.nll_loss(outputs, labels)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # show dataset
        if batch_idx == 0 and epoch == 0:
            fig = plt.figure()
            inputs = inputs.cpu()  # convert to cpu
            grid = utils.make_grid(inputs)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.show()

        # print loss and accuracy
        if(batch_idx+1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]  Loss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)  # must divide len(train_loader) here
    train_loss_lst.append(train_loss)
    train_acc_lst.append(correct / len(train_loader.dataset))
    return train_loss_lst, train_acc_lst


def test(model, test_loader, device, test_loss_lst, test_acc_lst):
    model.eval()  # Sets the module in evaluation mode
    test_loss = 0
    correct = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # add one batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # find index of max prob
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))

    test_loss_lst.append(test_loss)
    test_acc_lst.append(correct / len(test_loader.dataset))
    return test_loss_lst, test_acc_lst


if __name__ == "__main__":
    torch.manual_seed(0)
    # load datasets
    train_loader, test_loader = load_dataset()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # net = CIFAR10Net().to(device)
    net = CIFAR10LeNet().to(device)

    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train_loss_lst = []
    test_loss_lst = []
    train_acc_lst = []
    test_acc_lst = []
    for epoch in range(epochs):
        train_loss_lst, train_acc_lst = train(net, train_loader, optimizer,
                                              epoch, device, train_loss_lst, train_acc_lst)
        test_loss_lst, test_acc_lst = test(
            net, test_loader, device, test_loss_lst, test_acc_lst)

    # plot loss and accuracy
    fig = plt.figure()
    plt.plot(range(epochs), train_loss_lst, 'g', label='train loss')
    plt.plot(range(epochs), test_loss_lst, 'k', label='test loss')
    plt.plot(range(epochs), train_acc_lst, 'r', label='train acc')
    plt.plot(range(epochs), test_acc_lst, 'b', label='test acc')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('acc-loss')
    plt.legend(loc="upper right")
    now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()))
    plt.savefig('./parameter/' + now + '.jpg')
    plt.show()

    # save model
    torch.save(net, "./model/cifar10_cnn.pth")
