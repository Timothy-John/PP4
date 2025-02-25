import os
import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
from torchvision import transforms
import random
import PIL
import torch.nn.functional as F

class IndianCoverCQT(Dataset):
    def __init__(self, mode='train', model_name):
        self.indir = f'../CoverSongDetection_Timothy/Encodec/dataset/{model_name}'
        if mode=='train':
          self.filepath = 'data/coversIndian_train_val.txt'
        elif mode=='val':
          self.filepath = 'data/coversIndian_train_val.txt'
        else:
          self.filepath = 'data/coversIndian_test.txt'
        
        with open(self.filepath, 'r') as fp:
            self.file_list = [line.rstrip() for line in fp]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index].strip()
        set_id = filename.split('_')[0]  # Assuming the set_id is the first part before '_'
        in_path = os.path.join(self.indir, filename + '.npy')
        data = np.load(in_path)
        data = self.pad_or_truncate(data, 400, 84)
        return data, int(set_id)

    def pad_or_truncate(self, data, target_length):
        current_length = len(data)
        if current_length > target_length:
            data = data[:target_length,:,:]
        elif current_length < target_length:
            pad_time = target_length - current_length
            data = F.pad(data, (0, pad_time), mode='constant', value=0)
        return data
