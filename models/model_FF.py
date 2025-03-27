import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from collections import OrderedDict
import math
from .basic_module import BasicModule

class model_FF(BasicModule):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(1024, 600)
        self.fc1 = nn.Linear(600, 100)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x
