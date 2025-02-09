import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import models
from config import opt
from utility import *
from tqdm import tqdm
import numpy as np
from cqt_loader import IndianCover
import random

from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torchaudio.transforms as T
import librosa

#Setting Randomization Seed for Reproducibility
random.seed(7)
torch.manual_seed(7)

def custom_collate(batch):
    data = [item[0][0] for item in batch]
    sample_rate = [item[0][1] for item in batch]
    labels = [item[1] for item in batch]
    data = torch.stack(data)
    labels = torch.LongTensor(labels)
    sample_rate = torch.LongTensor(sample_rate)
    return (data, sample_rate), labels

def transfer_learning(**kwargs):
    opt.num_workers = 2
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt._parse(kwargs)
    
    test_data = IndianCover('test')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)#, collate_fn=custom_collate)

    model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True, device_map=opt.device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True, device_map=opt.device)
    
    test_map, test_top10, test_rank1 = val_slow(model, processor, test_loader, -1, "Indian Test Set")
    print(f"Final Test Set Performance - MAP: {test_map:.4f}, Top10: {test_top10:.4f}, Rank1: {test_rank1:.2f}")

@torch.no_grad()
def val_slow(model, processor, dataloader, epoch, dataset_name=None):
    model.eval()
    all_embeddings = []
    all_labels = []

    resample_rate = processor.sampling_rate
    for (data, sampling_rate), label in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
        # make sure the sample_rate aligned
        if resample_rate != sampling_rate:
            print(f'setting rate from {sampling_rate} to {resample_rate}')
            resampler = T.Resample(sampling_rate, resample_rate)
        else:
            resampler = None
        # audio file is decoded on the fly
        if resampler is None:
            input_audio = data[0]
        else:
            input_audio = resampler(torch.from_numpy(data))[0]
        inputs = processor(input_audio, sampling_rate=resample_rate, return_tensors="pt")

        inputs = inputs.to(opt.device)
        with torch.no_grad():
          outputs = model(**inputs, output_hidden_states=True)
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
        embeddings = all_layer_hidden_states.mean(-2)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(label.cpu().numpy())

    embeddings = np.concatenate(all_embeddings)
    labels = np.concatenate(all_labels)
    
    embeddings = norm(embeddings)
    dis2d = -np.matmul(embeddings, embeddings.T)
    MAP, top10, rank1 = calc_MAP(dis2d, labels)
    
    return MAP, top10, rank1

if __name__=='__main__':
    import fire
    fire.Fire()
