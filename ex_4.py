import sys
import torch
import numpy as np
import torchvision
from torch.utils import data
import torch.nn.functional as F
from models import ModelA, ModelB, ModelC, ModelD, ModelE, ModelF
import matplotlib.pyplot as plt

TRAIN_SIZE = 0.8
VALIDATION_SIZE = 0.2
IMAGE_SIZE = 784
EPOCH = 10


class MyDataSet(data.Dataset):
    def __init__(self, x, y=None, transform=None):
        self.__x = x
        self.__y = y
        self.__transform = transform

    def __len__(self):
        return len(self.__x)

    def __getitem__(self, index):
        x = self.__x[index]
        y = self.__y[index]
        x = np.asarray(x).astype(np.uint8).reshape(28, 28)
        if self.__transform:
            x = self.__transform(x)
        if self.__y is not None:
            return x, y
        return x


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


def show_graph_accuracy(loss_train, loss_val):
    pass


def show_graph_loss(loss_train, loss_val):
    lt = np.array(loss_train)
    lv = np.array(loss_val)
    epoch = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.plot(epoch, lt, 'g', label='Training loss')
    plt.plot(epoch, lv, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def run_model(model, train_loader, validation_loader):
    loss_trains = []
    loss_tests = []
    for e in range(1, EPOCH + 1):
        loss_trains.append(train(model, train_loader))
        loss_tests.append(test(model, validation_loader))
    show_graph_loss(loss_trains, loss_tests)


def train(model, train_loader):
    model.train()
    losses = 0
    for batch_idx, (data_, labels) in enumerate(train_loader):
        model.optimizer.zero_grad()
        output = model(data_)
        loss = F.nll_loss(output, labels)
        losses += F.nll_loss(output, labels, size_average=False).item()
        loss.backward()
        model.optimizer.step()
    return losses


def test(model, validation_loader):
    model.eval()
    test_loss = 0
    tmp_loss = 0
    correct = 0
    with torch.no_grad():
        for data_, target in validation_loader:
            output = model(data_)
            # sum up batch loss
            tmp_loss += F.nll_loss(output, target, size_average=False).item()
            test_loss += F.nll_loss(output, target, reduction="mean").item()
            # get index of the max log - probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    test_loss /= len(validation_loader.dataset)
    print("\nTests set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(test_loss, correct,
                                                                                  len(validation_loader.dataset),
                                                                                  100. * correct / len(
                                                                                      validation_loader.sampler)))
    return tmp_loss


def create_loaders(train_x, train_y, validation_x, validation_y):
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_set = MyDataSet(train_x, train_y, transforms)
    validation_set = MyDataSet(validation_x, validation_y, transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=True)
    return train_loader, validation_loader


def run_models(train_loader, validation_loader):
    print("*********************** MODEL A ******************** ")
    model_a = ModelA(IMAGE_SIZE)
    run_model(model_a, train_loader, validation_loader)
    print("******************** MODEL B *********************** ")
    model_b = ModelB(IMAGE_SIZE)
    run_model(model_b, train_loader, validation_loader)
    print("******************** MODEL C *********************** ")
    model_c = ModelC(IMAGE_SIZE)
    run_model(model_c, train_loader, validation_loader)
    print("******************** MODEL D *********************** ")
    model_d = ModelD(IMAGE_SIZE)
    run_model(model_d, train_loader, validation_loader)
    print("******************** MODEL E *********************** ")
    model_e = ModelE(IMAGE_SIZE)
    run_model(model_e, train_loader, validation_loader)
    print("******************** MODEL F *********************** ")
    model_f = ModelF(IMAGE_SIZE)
    run_model(model_f, train_loader, validation_loader)


def main():
    try:
        train_x = np.loadtxt(sys.argv[1])
        train_y = np.loadtxt(sys.argv[2], dtype="int64")
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
