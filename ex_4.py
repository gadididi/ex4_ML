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
        x = np.asarray(x).astype(np.uint8).reshape(28, 28)
        if self.__transform:
            x = self.__transform(x)
        if self.__y is not None:
            y = self.__y[index]
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


def show_graph(train_res, test_res, what_kind):
    lt = np.array(train_res)
    lv = np.array(test_res)
    epoch = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.plot(epoch, lt, 'g', label='Training {}'.format(what_kind))
    plt.plot(epoch, lv, 'b', label='validation {}'.format(what_kind))
    plt.title('Training and Validation {}'.format(what_kind))
    plt.xlabel('Epochs')
    plt.ylabel(what_kind)
    plt.legend()
    plt.show()


def run_model(model, train_loader, test_loader, test_x_loaders):
    loss_trains = []
    loss_tests = []
    accuracy_train = []
    accuracy_test = []
    for e in range(1, EPOCH + 1):
        loss_tr, acc_tr = train(model, train_loader)
        loss_te, acc_te = test(model, test_loader)
        loss_trains.append(loss_tr / len(train_loader.sampler))
        loss_tests.append(loss_te / len(test_loader.sampler))
        accuracy_train.append(acc_tr)
        accuracy_test.append(acc_te)
    # show_graph(loss_trains, loss_tests, 'Loss')
    # show_graph(accuracy_train, accuracy_test, 'Accuracy')
    predict(model, test_x_loaders)


def train(model, train_loader):
    model.train()
    losses = 0
    correct = 0
    for batch_idx, (data_, labels) in enumerate(train_loader):
        model.optimizer.zero_grad()
        output = model(data_)
        loss = F.nll_loss(output, labels)
        losses += F.nll_loss(output, labels, reduction="mean").item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).cpu().sum()
        loss.backward()
        model.optimizer.step()
    return losses, (100. * correct / len(train_loader.sampler))


def test(model, test_loader):
    model.eval()
    test_loss = 0
    tmp_loss = 0
    correct = 0
    with torch.no_grad():
        for data_, target in test_loader:
            output = model(data_)
            # sum up batch loss
            tmp_loss += F.nll_loss(output, target, reduction="mean").item()
            test_loss += F.nll_loss(output, target, reduction="mean").item()
            # get index of the max log - probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print("\nTests set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(test_loss, correct,
                                                                                  len(test_loader.dataset),
                                                                                  100. * correct / len(
                                                                                      test_loader.sampler)))
    return tmp_loss, (100. * correct / len(test_loader.sampler))


def create_loaders(train_x, train_y, validation_x=None, validation_y=None):
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_set = MyDataSet(train_x, train_y, transforms)
    validation_loader = None
    if validation_x is not None:
        validation_set = MyDataSet(validation_x, validation_y, transforms)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_set)

    return train_loader, validation_loader


def run_models(train_loader, validation_loader, test_x_loaders):
    print("******************** MODEL D *********************** ")
    model_b = ModelD(IMAGE_SIZE)
    run_model(model_b, train_loader, validation_loader, test_x_loaders)


def create_original_loaders():
    train_set = torchvision.datasets. \
        FashionMNIST(root="./data", download=True, transform=torchvision.transforms.ToTensor())
    test_set = torchvision.datasets. \
        FashionMNIST(root="./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
    return train_loader, test_loader


def predict(best_model, test_x_loaders):
    test_y = open("test_y", 'w+')
    best_model.eval()
    with torch.no_grad():
        for x in test_x_loaders:
            output = best_model(x)
            pred = output.data.max(1, keepdim=True)[1].item()
            line = str(pred) + "\n"
            test_y.write(line)
    test_y.close()


def main():
    try:
        train_x = np.loadtxt(sys.argv[1])
        train_y = np.loadtxt(sys.argv[2], dtype="int64")
        test_x = np.loadtxt(sys.argv[3])
        test_x_loaders, _ = create_loaders(test_x, None, None, None)
        train_x, train_y, validation_x, validation_y = split_validation_train(train_x, train_y)
        train_loader, validation_loader = create_loaders(train_x, train_y, validation_x, validation_y)
        # train_fashion_loader, test_fashion_loader = create_original_loaders()
        run_models(train_loader, validation_loader, test_x_loaders)
    except IndexError as e:
        print(e)
        exit(-1)
    return


if __name__ == '__main__':
    main()
