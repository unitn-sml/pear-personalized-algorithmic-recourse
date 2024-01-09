import torch.nn as nn
import torch.nn.functional as F

class NetLinear(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear(dimensions, 1)

    def forward(self, x):
        if not self.training:
            x = self.sigmoid(self.output(x))
        else:
            # The sigmoid is applied directly by the loss
            x = self.output(x)
        return x

class Net(nn.Module):
    def __init__(self, dimensions, layers=4):
        super().__init__()

        self.linears = nn.ModuleList([
            nn.Linear(dimensions//i, dimensions//(i+1))
            for i in range(1, layers)
        ])
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear(dimensions//layers, 1)

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = F.relu(l(x))
        if not self.training:
            x = self.sigmoid(self.output(x))
        else:
            x = self.output(x)
        return x