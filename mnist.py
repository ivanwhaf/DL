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

batch_size = 64
epochs = 50


class Net(nn.Module):
    """
    customized neural network
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # block1
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.relu(out)

        # block2
        out = self.conv2(out)
        out = self.maxpool(out)
        out = self.relu(out)

        # block3
        out = self.conv3(out)
        out = self.relu(out)

        # block4
        out = out.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)

        return out


def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        './dataset', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(
        './dataset', train=False, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(model, train_loader, optimizer, epoch, device, train_loss_lst, train_acc_lst):
    model.train()  # Sets the module in training mode
    correct = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        pred = outputs.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()

        loss = F.nll_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        # show dataset
        if batch_idx == 0 and epoch == 0:
            fig = plt.figure()
            inputs = inputs.cpu()  # convert to cpu
            grid = utils.make_grid(inputs)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.show()

        # print loss and accuracy
        if(batch_idx+1) % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]  Loss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))

    train_loss_lst.append(loss.item())
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = Net().to(device)

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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
    torch.save(net, "./model/mnist_cnn.pth")
