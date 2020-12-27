import sys
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torch.utils import data
import torch.nn.functional as F

TRAIN_SIZE = 0.8
VALIDATION_SIZE = 0.2
IMAGE_SIZE = 784
EPOCH = 10


class MyDataSet(data.Dataset):
    def __init__(self, x, y, transform=None):
        self.__x = torch.from_numpy(x).float()
        self.__y = torch.from_numpy(y).float()
        self.__transform = transform

    def __len__(self):
        return len(self.__x)

    def __getitem__(self, index):
        x = self.__x[index]
        y = self.__y[index]

        if self.__transform:
            x = self.__transform(x)
        return x, y


class ModelA(nn.Module):
    def __init__(self, image_size):
        super(ModelA, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)


def split_validation_train(train_x, train_y):
    # shuffle the data set
    indices = np.arange(train_x.shape[0])
    np.random.shuffle(indices)
    train_x = train_x[indices]
    train_y = train_y[indices]
    validation_size = int(len(train_x) * VALIDATION_SIZE)
    train_x = train_x[validation_size:]
    train_y = train_y[validation_size:]
    validation_x = train_x[:validation_size]
    validation_y = train_y[:validation_size]
    return train_x, train_y, validation_x, validation_y


def train(model, train_loader):
    model.train()
    for batch_idx, (data_, labels) in enumerate(train_loader):
        model.optimizer.zero_grad()
        output = model(data_)
        loss = F.nll_loss(output, labels)
        loss.backward()
        model.optimizer.step()


def test(model, validation_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data_, target in validation_loader:
            output = model(data_)


def create_loaders(train_x, train_y, validation_x, validation_y):
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_set = MyDataSet(train_x, train_y, transforms)
    validation_set = MyDataSet(validation_x, validation_y, transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=True)
    return train_loader, validation_loader


def run_models(train_loader, validation_loader):
    model_a = ModelA(IMAGE_SIZE)
    train(model_a, train_loader)


def main():
    try:

        train_x = np.loadtxt(sys.argv[1])
        train_y = np.loadtxt(sys.argv[2], dtype='int')
        test_x = np.loadtxt(sys.argv[3])
        train_x, train_y, validation_x, validation_y = split_validation_train(train_x, train_y)
        train_loader, validation_loader = create_loaders(train_x, train_y, validation_x, validation_y)
        run_models(train_loader, validation_loader)
    except IndexError as e:
        print(e)
        exit(-1)
    return


if __name__ == '__main__':
    main()
