import random

from model import *


class BPVNet(nn.Module):
    def __init__(self, num_class):
        super(BPVNet, self).__init__()
        self.f_Net1 = fNet()
        self.f_Net2 = fNet()

        self.fc1 = nn.Linear(512*5, num_class)
        self.fc2 = nn.Linear(512*5, num_class)

    def forward(self, x1, x2, x3, x4, x5):
        random_numbers = random.sample(range(1, 6), 2)
        random_number1 = 6
        random_number2 = random_numbers[1]
        x_1 = self.f_Net1(x1, x2, x3, x4, x5,random_number1)
        x_2 = self.f_Net2(x1, x2, x3, x4, x5,random_number2)

        x_1 = self.fc1(x_1)
        x_2 = self.fc2(x_2)

        return x_1, x_2
