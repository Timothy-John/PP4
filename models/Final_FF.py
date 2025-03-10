import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from collections import OrderedDict
import math
from .basic_module import BasicModule

class Final_FF(BasicModule):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(400, 300)
        self.fc1 = nn.Linear(300, 300)

    def forward(self, x,MERT_FF_out):
        x = torch.cat([x.squeeze(),MERT_FF_out.squeeze()]).unsqueeze(0)
        feat = self.fc0(x)
        x = self.fc1(feat)
        return x,feat
