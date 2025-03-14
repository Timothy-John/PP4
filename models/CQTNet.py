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
        self.fc0 = nn.Linear(1024, 300)
        self.fc1 = nn.Linear(300, 300)

    def forward(self, x):
        N = x.size()[0]
        x = x.view(N, -1)
        feature = self.fc0(x)
        x = self.fc1(feature)
        return x, feature
