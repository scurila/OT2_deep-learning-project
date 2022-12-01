import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 6 * 6, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

        # Model 12, adding a further fully connected layer
        # self.fc1 = nn.Linear(16 * 6 * 6, 32)
        # self.fc2 = nn.Linear(32, 16)
        # self.fc3 = nn.Linear(16, 8)
        # self.fc4 = nn.Linear(8, 2)

        # self.fc1 = nn.Linear(16 * 6 * 6, 64)
        # self.fc2 = nn.Linear(64, 32)
        # self.fc3 = nn.Linear(32, 2)

        # self.fc1 = nn.Linear(16 * 6 * 6, 128)
        # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 6 * 6)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Adding sigmoid
        # x = torch.sigmoid(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))

        # x = F.relu(self.fc3(x))

        x = self.fc3(x)
        # x = self.fc4(x)
        
        # m = nn.Softmax(dim=1)
        # x = m(self.fc3(x))
        return x


