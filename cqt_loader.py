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
    def __init__(self, mode='train', out_length=None):
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
        filename1 = self.file_list[index].strip()
        if index+1 >= len(self.file_list):
            filename2 = self.file_list[0].strip()
        else:
            filename2 = self.file_list[index+1].strip()
        
        set_id1 = filename1.split('_')[0]  # Assuming the set_id is the first part before '_'
        set_id2 = filename2.split('_')[0]
        if set_id1==set_id2:
            label = 1
        else:
            label = 0

        in_path1 = os.path.join(self.indir, filename1 + '.npy')
        in_path2 = os.path.join(self.indir, filename2 + '.npy')
        data1 = np.load(in_path1)
        data2 = np.load(in_path2)
        
        transform_test = transforms.Compose([
            lambda x: x.astype(np.float32) / (np.max(np.abs(x)) + 1e-6),
            lambda x: self.cut_data_front(x, self.out_length),
            lambda x: torch.Tensor(x),
            lambda x: x.unsqueeze(0),  # Add channel dimension
        ])
        
        data1 = transform_test(data1)
        data1 = self.pad_or_truncate(data1, 400, 84)
        data2 = transform_test(data2)
        data2 = self.pad_or_truncate(data2, 400, 84)
        
        return (data1, data2), float(label)

    # Including necessary methods from the CQT class
    def pad_or_truncate(self, data, target_length, target_freq=84):
        if data.ndim == 2:
            data = data.unsqueeze(0)
        
        _, freq, current_length = data.shape
        
        if freq > target_freq:
            data = data[:, :target_freq, :]
        elif freq < target_freq:
            pad_freq = target_freq - freq
            data = F.pad(data, (0, 0, 0, pad_freq), mode='constant', value=0)
        
        if current_length > target_length:
            data = data[:, :, :target_length]
        elif current_length < target_length:
            pad_time = target_length - current_length
            data = F.pad(data, (0, pad_time), mode='constant', value=0)
        
        return data

    def cut_data_front(self, data, out_length):
        if out_length is not None:
            if data.shape[1] > out_length:
                data = data[:, :out_length]
            else:
                offset = out_length - data.shape[1]
                data = np.pad(data, ((0, 0), (0, offset)), "constant")
        if data.shape[1] < 200:
            offset = 200 - data.shape[1]
            data = np.pad(data, ((0, 0), (0, offset)), "constant")
        return data

if __name__ == '__main__':
    train_dataset = IndianCoverCQT('train', 394)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=12, shuffle=True)
