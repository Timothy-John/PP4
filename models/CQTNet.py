import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from collections import OrderedDict
import math
from .basic_module import BasicModule

class CQTNet(BasicModule):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(768, 500)
        self.fc1 = nn.Linear(500, 300)
        self.fc2 = nn.Linear(300, 300)

    def forward(self, x):
        N = x.size()[0]
        x = x.view(N, -1)
        x = self.fc0(x)
        feature = self.fc1(x)
        x = self.fc2(feature)
        return x, feature
