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
        self.model = model
        if self.model=='MERT':
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
        if self.model=='CQTNet':
            in_path = os.path.join(self.indir, filename + '.npy')
            data = np.load(in_path)
            transform_test = transforms.Compose([
                lambda x: x.astype(np.float32) / (np.max(np.abs(x)) + 1e-6),
                lambda x: self.cut_data_front(x, self.out_length),
                lambda x: torch.Tensor(x),
                lambda x: x.unsqueeze(0),  # Add channel dimension
            ])
            data = transform_test(data)
            data = self.pad_or_truncate(data, 768, 84) #768 is the embedding size of MERT
        else:
            in_path = self.indir +filename[:-int(len(filename.split('_')[-1])+1)] +'/' +filename +'.mp3'
            data, sr = librosa.load(in_path, sr=24000)
        return data, int(set_id)
