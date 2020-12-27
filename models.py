import torch.nn as nn
import torch
import torch.nn.functional as F

LR = 0.09


class ModelA(nn.Module):
    def __init__(self, image_size):
        super(ModelA, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.02)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ModelB(nn.Module):
    def __init__(self, image_size):
        super(ModelB, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ModelC(nn.Module):
    def __init__(self, image_size):
        super(ModelC, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.05)

    def forward(self, x):
        drop = nn.Dropout(p=0.2)
        x = x.view(-1, self.image_size)
        x = drop(x)
        x = F.relu(self.fc0(x))
        x = drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ModelD(nn.Module):
    def __init__(self, image_size):
        super(ModelD, self).__init__()
        self.image_size = image_size
        self.classifier = nn.Sequential(
            nn.Linear(image_size, 100),
            nn.BatchNorm1d(100),  # applying batch norm
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 10))
        self.optimizer = torch.optim.SGD(self.parameters(), lr=LR)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class ModelE(nn.Module):
    def __init__(self, image_size):
        super(ModelE, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=LR)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(self.fc5(x))


class ModelF(nn.Module):
    def __init__(self, image_size):
        super(ModelF, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=LR)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.sigmoid(self.fc0(x))
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(self.fc5(x))
