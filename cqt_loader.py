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
    def __init__(self, mode='train', model='CQTNet', out_length=None):
        if model=='MERT':
            self.indir = '/content/drive/MyDrive/CoverSongDetection_Timothy/CoverIndian_audio/'
        else:
            self.indir = '/content/drive/MyDrive/CoverSongDetection_Timothy/IndianCover_cqt_npy'
        
        if mode=='train':
          self.filepath = 'data/coversIndian_train_val.txt'
        elif mode=='val':
          self.filepath = 'data/coversIndian_train_val.txt'
        else:
          self.filepath = 'data/coversIndian_test.txt'
        
        with open(self.filepath, 'r') as fp:
            self.file_list = [line.rstrip() for line in fp]
        self.out_length = out_length

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index].strip()
        set_id = filename.split('_')[0]  # Assuming the set_id is the first part before '_'
        in_path = self.indir +filename[:-int(len(filename.split('_')[-1])+1)] +'/' +filename +'.mp3'
        data, sr = librosa.load(in_path, sr=24000)
        return data, int(set_id)
