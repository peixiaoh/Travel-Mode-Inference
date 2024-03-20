import os
import pickle
import random
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils import data
from torchinfo import summary

from pytorchtools import EarlyStopping
from utils import multi_scale_eca_resnet


def setup_seed(seed=None, workers=False):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed


def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()

    sum_loss = 0
    correct = 0
    total_num = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)

        pred = output.data.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print_loss = loss.data.item()
        sum_loss += print_loss
        if (batch_idx + 1) % 10 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.data.item()
            ))

    avg_loss = sum_loss / len(train_loader)
    acc = correct / total_num
    print("epoch: {}, loss: {:.4f}, accuracy: {:.1f}".format(epoch, avg_loss, 100. * acc))

    return acc, avg_loss


def val(validation_loader, model, criterion, device):
    model.eval()

    test_loss = 0
    correct = 0
    total_num = len(validation_loader.dataset)
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            print_loss = loss.item()
            test_loss += print_loss

    avg_loss = test_loss / len(validation_loader)
    acc = correct / total_num
    print("\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)".format(
        avg_loss, correct, len(validation_loader.dataset), 100. * acc
    ))

    return acc, avg_loss


def test(test_loader, model, device):
    model.eval()

    y_score, y_true = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            y_score.append(F.softmax(output, dim=1).cpu().numpy())
            y_true.append(target.cpu().numpy())

    y_score = np.concatenate(y_score, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    return y_score, y_true


def adjust_learning_rate(optimizer, epoch):
    lr_new = 0.001 * (0.1 ** (epoch // 20))
    print("lr:", lr_new)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_new


def read_data(filename='data/Revised_KerasData_Smoothing.pickle', random_state=0):
    # load data
    with open(filename, mode='rb') as f:
        TotalInput, FinalLabel = pickle.load(f, encoding='latin1')  # Also can use the encoding 'iso-8859-1'
    TotalInput = np.transpose(np.squeeze(TotalInput, axis=1), (0, 2, 1))

    # Divide the dataset into training dataset, validation dataset and test dataset
    X_train, X_test, y_train, y_test = \
        train_test_split(TotalInput, FinalLabel, test_size=0.2, random_state=random_state)
    X_train, X_validation, y_train, y_validation = \
        train_test_split(X_train, y_train, test_size=0.1, random_state=random_state)

    # numpy --> tensor
    X_train, y_train = \
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64)
    X_validation, y_validation = \
        torch.tensor(X_validation, dtype=torch.float32), torch.tensor(y_validation, dtype=torch.int64)
    X_test, y_test = \
        torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int64)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def main():
    random_state = 0
    setup_seed(random_state)

    batch_size = 64
    epochs = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "../saved_model/model_msecaresnet-best.pth"

    X_train, X_validation, X_test, y_train, y_validation, y_test = read_data(random_state=random_state)

    dataset_train = data.TensorDataset(X_train, y_train)
    dataset_validation = data.TensorDataset(X_validation, y_validation)
    dataset_test = data.TensorDataset(X_test, y_test)

    train_loader = data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    validation_loader = data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=False)
    test_loader = data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    model = multi_scale_eca_resnet.MsEcaResNet(input_channel=4, layers=[1, 1, 1, 1], num_classes=5)
    summary(model, input_size=(batch_size, 4, 300))
    model.to(device)


    from thop import profile

    inputs = torch.randn(1, 4, 300).to(device)
    flops, params = profile(model, inputs=(inputs,))
    print('FLOPs = {}M'.format(flops / 1e6))
    print('Params = {}M'.format(params / 1e6))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=10, verbose=True, path=model_path)

    accs, losses = [], []
    val_accs, val_losses = [], []
    for epoch in range(1, epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        acc, loss = train(train_loader, model, criterion, optimizer, epoch, device)
        val_acc, val_loss = val(validation_loader, model, criterion, device)
        early_stopping(val_loss, model)

        accs.append(acc)
        losses.append(loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(model_path))

    y_score, y_true = test(test_loader, model, device)
    y_pred = np.argmax(y_score, axis=1)

    print("Test set:")
    print(classification_report(y_true, y_pred, target_names=["walk", "bike", "bus", "Driving", "train"], digits=3))
    pprint(confusion_matrix(y_true, y_pred))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].plot(losses, label="loss")
    axes[0].plot(val_losses, label="val_loss")
    axes[0].set(xlabel="Epoch", title="Loss")
    axes[0].legend(loc="upper right")

    axes[1].plot(accs, label="acc")
    axes[1].plot(val_accs, label="val_acc")
    axes[1].set(xlabel="Epoch", title="Accuracy")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
