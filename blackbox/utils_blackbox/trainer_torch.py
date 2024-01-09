from .dataloader import Data
from .net import Net, NetLinear

import torch
import torch.optim as optim

from sklearn.metrics import f1_score, accuracy_score, recall_score

import numpy as np

def train_nn(train_X, train_y, validation_X, validation_y, linear=False, iterations=100, layers=4, batch_size=1024):

    train = Data(train_X, train_y)
    validation = Data(validation_X, validation_y)

    if linear:
        net = NetLinear(train.feature_size())
    else:
        net = Net(train.feature_size(), layers)

    print(net)
    print("Train size: ", len(train))
    print("Feature size: ", train.feature_size())

    class_sample_count = np.array(
        [len(np.where(train_y == t)[0]) for t in np.unique(train_y)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in train_y])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight.double(), len(samples_weight))

    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                              num_workers=2,
                                              sampler=sampler)

    valloader = torch.utils.data.DataLoader(validation, batch_size=batch_size,
                                            num_workers=2)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in range(iterations):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            labels = labels.view((len(labels), 1))

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 2000 mini-batches

                Y_pred = []
                Y_true = []

                net.train(False)
                for _, val in enumerate(valloader, 0):

                    inputs, labels = val

                    with torch.no_grad():
                        val_out = torch.round(net(inputs)).flatten()

                    Y_true += labels.numpy().flatten().tolist()
                    Y_pred += val_out.numpy().flatten().tolist()

                net.train(True)
                print('[%d, %5d] loss: %.3f / f1: %.3f, acc: %.3f, rec: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10,
                      f1_score(Y_pred, Y_true, average="macro"),
                      accuracy_score(Y_pred, Y_true),
                      recall_score(Y_pred, Y_true)
                      ))
                running_loss = 0.0

    return net
    