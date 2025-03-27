import os
import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
from torchvision import transforms
import random
import PIL
import torch.nn.functional as F

class IndianCover(Dataset):
    def __init__(self, mode='train', out_length=None):
        self.mode = mode
        if mode=='train':
          self.filepath = 'data/coversIndian_augmented_train_val.txt'
          self.set = 0
          self.set_dict = {}
          self.filepath = 'data/coversIndian_train_val.txt'
          self.indir = '/content/drive/MyDrive/CoverSongDetection_Timothy/330M_MERT_train_embeddings.npy'
        elif mode=='val':
          self.filepath = 'data/coversIndian_train_val.txt'
          self.indir = '/content/drive/MyDrive/CoverSongDetection_Timothy/330M_MERT_train_embeddings.npy'
        else:
          self.filepath = 'data/coversIndian_test.txt'
          self.indir = '/content/drive/MyDrive/CoverSongDetection_Timothy/330M_MERT_test_embeddings.npy'
        
        with open(self.filepath, 'r') as fp:
            self.file_list = [line.rstrip() for line in fp]
        self.out_length = out_length

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index].strip()
        set_id = filename.split('_')[0]  # Assuming the set_id is the first part before '_'
        if self.mode == 'train':
            if set_id in self.set_dict:
                set_id = self.set_dict[set_id]
            else:
                self.set_dict[set_id] = self.set
                set_id = self.set
                self.set += 1
        data = np.load(self.indir)[index]
        return data, int(set_id)
        
