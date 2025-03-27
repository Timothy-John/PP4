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
import librosa
import random

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def custom_collate(batch):
    data = [item[0][0] for item in batch]
    labels = [item[1] for item in batch]
    return torch.Tensor(np.asarray(data)), torch.LongTensor(np.asarray(labels))

def transfer_learning(**kwargs):
    opt._parse(kwargs)
    opt.batch_size = 32
    opt.num_workers = 2
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {opt.device}")
    
    train_data = IndianCover('train')
    val_data = IndianCover('val')
    test_data = IndianCover('test')

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=custom_collate)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)

    #MERT Embeddings already loaded in GDrive
    #MERTmodel = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True, device_map=opt.device)
    #MERTprocessor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True, device_map=opt.device)

    NNmodel = getattr(models, 'CQTNet')()
    NNmodel = NNmodel.to(opt.device)

    for name, param in NNmodel.named_parameters():
        param.requires_grad = True
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, NNmodel.parameters()), lr=1e-4)
    
    # Training loop
    opt.max_epoch = 200
    best_val_map = 0
    best_model_path = None
    
    for epoch in range(opt.max_epoch):
        #MERTmodel.eval()
        NNmodel.train()
        total_loss = 0
        
        for MERTembeddings, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.max_epoch}"):
            # make sure the sample_rate aligned
            #inputs = MERTprocessor(data, sampling_rate=24000, return_tensors="pt", padding=True).to(opt.device)
            #labels = torch.LongTensor(labels).to(opt.device)

            #with torch.no_grad():
            #    MERTembeddings = MERTmodel(**inputs, output_hidden_states=False)
            optimizer.zero_grad()
            scores, _ = NNmodel(MERTembeddings.to(opt.device))
            
            loss = criterion(scores, labels.to(opt.device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{opt.max_epoch}, Loss: {avg_loss:.4f}")

        # Evaluate on validation set
        val_map, val_top10, val_rank1 = val_slow(NNmodel, val_loader, epoch, "Indian Validation Set")
        print(f"Validation - MAP: {val_map:.4f}, Top10: {val_top10:.4f}, Rank1: {val_rank1:.2f}")

        if val_map > best_val_map:
            best_val_map = val_map
            best_model_path = f"check_points/MERT_transfer_learning_epoch_{epoch+1}.pth"
            torch.save(NNmodel.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
    
    # Load best model and evaluate on test set
    NNmodel.load_state_dict(torch.load(best_model_path))
    test_map, test_top10, test_rank1 = val_slow(NNmodel, test_loader, -1, "Indian Test Set")
    print(f"Final Test Set Performance - MAP: {test_map:.4f}, Top10: {test_top10:.4f}, Rank1: {test_rank1:.2f}")

@torch.no_grad()
def val_slow(NNmodel, dataloader, epoch, dataset_name=None):
    #MERTmodel.eval()
    NNmodel.eval()
    all_embeddings = []
    all_labels = []
    
    for data, label in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
        #inputs = MERTprocessor(data, sampling_rate=24000, return_tensors="pt")

        #inputs = inputs.to(opt.device)
        with torch.no_grad():
        #  MERTembeddings = MERTmodel(**inputs, output_hidden_states=False)
          _, embeddings = NNmodel(data.to(opt.device))
        
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(label)
    
    embeddings = np.concatenate(all_embeddings)
    labels = np.concatenate(all_labels)
    
    embeddings = norm(embeddings)
    dis2d = -np.matmul(embeddings, embeddings.T)
    MAP, top10, rank1 = calc_MAP(dis2d, labels)
    
    return MAP, top10, rank1

if __name__=='__main__':
    import fire
    fire.Fire()
