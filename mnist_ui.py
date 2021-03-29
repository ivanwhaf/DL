import tkinter as tk

import torch
from PIL import Image, ImageDraw
from torch import nn
from torchvision import transforms


# import numpy as np


class Window:
    def __init__(self):
        window = tk.Tk()
        window.title('MNIST')
        window.geometry('500x300')  # window size
        window['bg'] = 'gray'  # background color
        window.attributes("-alpha", 0.95)  # diaphaneity

        self.model_init()

        label = tk.Label(window, text='请在画板上画出数字，并点击“识别”按钮识别数字', font=('Arial', 10))
        label.pack()

        # hand-drawing canvas
        self.canvas = tk.Canvas(window, bg='black', width=200, height=200)
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self.onLeftButtonDown)
        self.canvas.bind('<B1-Motion>', self.onLeftButtonMove)
        self.canvas.bind('<ButtonRelease-1>', self.onLeftButtonUp)

        # button recognize
        btn_p = tk.Button(window, text="识别", font=('Arial', 10), width=10, height=2,
                          command=lambda: self.predict(self.canvas, window))
        btn_p.pack()
        btn_p.place(x=100, y=230)

        # button clear
        btn_c = tk.Button(window, text="清空", font=(
            'Arial', 10), width=10, height=2, command=self.clear)
        btn_c.pack()
        btn_c.place(x=310, y=230)

        # PIL image draw
        self.image = Image.new("RGB", (200, 200), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)

        # bottom status bar
        self.statusbar = tk.Label(
            window, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        window.mainloop()

    def model_init(self):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cuda')
        # net = Net().to(device)
        self.model = torch.load('model/mnist_cnn.pth')
        self.model.to(self.device)
        self.model.eval()
        print('model loading complete.')

    def onLeftButtonDown(self, event):
        self.lastx, self.lasty = event.x, event.y
        self.statusbar.config(text='btn down')
        print('btn down...')

    def onLeftButtonMove(self, event):
        self.canvas.create_line(self.lastx, self.lasty,
                                event.x, event.y, fill='white', width=8)
        self.draw.line([self.lastx, self.lasty, event.x,
                        event.y], (255, 255, 255), width=10)
        self.lastx, self.lasty = event.x, event.y
        self.statusbar.config(text='x:{}, y:{}'.format(event.x, event.y))
        print(event.x, event.y)

    def onLeftButtonUp(self, event):
        self.lastx, self.lasty = 0, 0
        print('btn up')

    def predict(self, canvas, window):
        # self.image.save('canvas.jpg')
        image = self.image.resize((28, 28))
        image = image.convert('L')
        # image = np.expand_dims(image, 0)
        # image = np.array(image)
        # image = torch.Tensor(image)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image = transform(image)
        # image = Variable(torch.unsqueeze(
        #     image, dim=0).float(), requires_grad=False)
        image.unsqueeze_(0)
        image = image.to(self.device)
        output = self.model(image)
        # find index of max prob
        pred = output.max(1, keepdim=True)[1]
        num = pred.cpu().numpy()[0][0]
        self.statusbar.config(text='predict num:' + str(num))
        print('predict:' + str(num))

    def clear(self):
        # clear canvas
        self.canvas.delete(tk.ALL)
        self.image = Image.new("RGB", (200, 200), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)


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
        self.softmax = nn.LogSoftmax(dim=1)  # log to solve value overflow

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


if __name__ == "__main__":
    window = Window()
