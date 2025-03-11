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


def custom_collate_MERT(batch):
    data = [item[0][0] for item in batch]
    labels = [item[1] for item in batch]
    labels = torch.LongTensor(labels)
    return torch.Tensor(np.asarray(data)), labels
def custom_collate_CQTNet(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    data = torch.stack(data)
    labels = torch.LongTensor(labels)
    return data, labels

def transfer_learning(**kwargs):
    opt._parse(kwargs)
    opt.batch_size = 1  #Do Not Change! Supports only BS 1
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {opt.device}")
    
    train_CQTNet_data = IndianCover('train', 'CQTNet')
    train_MERT_data = IndianCover('train', 'MERT')
    val_CQTNet_data = IndianCover('val', 'CQTNet')
    val_MERT_data = IndianCover('val', 'MERT')
    test_CQTNet_data = IndianCover('test', 'CQTNet')
    test_MERT_data = IndianCover('test', 'MERT')

    train_CQTNet_loader = DataLoader(train_CQTNet_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=custom_collate_CQTNet)
    train_MERT_loader = DataLoader(train_MERT_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=custom_collate_MERT)
    val_CQTNet_loader = DataLoader(val_CQTNet_data, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate_CQTNet)
    val_MERT_loader = DataLoader(val_MERT_data, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate_MERT)
    test_CQTNet_loader = DataLoader(test_CQTNet_data, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate_CQTNet)
    test_MERT_loader = DataLoader(test_MERT_data, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate_MERT)

    opt.model = 'CQTNet'
    opt.load_model_path = '../CoverSongDetection_Timothy/CQTNet_SpecAugment_x3.pth'
    CQTNet_model = getattr(models, 'CQTNet')()
    CQTNet_model.load(opt.load_model_path)
    CQTNet_model = CQTNet_model.to(opt.device)

    MERT_FF = getattr(models, 'MERT_FF')().to(opt.device)

    Final_FF = getattr(models, 'Final_FF')().to(opt.device)

    for name, param in CQTNet_model.named_parameters():
        param.requires_grad = True
    for name, param in MERT_FF.named_parameters():
        param.requires_grad = True
    for name, param in Final_FF.named_parameters():
        param.requires_grad = True
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, CQTNet_model.parameters()), lr=1e-4)

    # Training loop
    opt.max_epoch = 50
    best_val_map = 0
    best_CQTNet_path = None
    best_MERT_FF_path = None
    best_Final_FF_path = None
    
    optimizer.zero_grad()
    for epoch in range(opt.max_epoch):
        CQTNet_model.train()
        MERT_FF.train()
        Final_FF.train()
        total_loss = 0
        
        for i, ((MERT_data, MERT_labels), (CQTNet_data, CQTNet_labels)) in enumerate(tqdm(zip(train_MERT_loader,train_CQTNet_loader), desc=f"Epoch {epoch+1}/{opt.max_epoch}")):
            assert MERT_labels == CQTNet_labels
            
            labels = torch.LongTensor(CQTNet_labels).to(opt.device)

            optimizer.zero_grad()
            MERT_FF_out = MERT_FF(MERT_data.to(opt.device))
            feat = CQTNet_model(CQTNet_data.to(opt.device))
            scores,_ = Final_FF(feat,MERT_FF_out)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_CQTNet_loader)
        print(f"Epoch {epoch+1}/{opt.max_epoch}, Loss: {avg_loss:.4f}")

        # Evaluate on validation set
        val_map, val_top10, val_rank1 = val_slow(CQTNet_model, MERT_FF, Final_FF, val_CQTNet_loader, val_MERT_loader, epoch, "Indian Validation Set")
        print(f"Validation - MAP: {val_map:.4f}, Top10: {val_top10:.4f}, Rank1: {val_rank1:.2f}")

        if val_map > best_val_map:
            best_val_map = val_map
            best_CQTNet_path = f"check_points/CQTNet_transfer_learning_epoch_{epoch+1}.pth"
            best_MERT_FF_path = f"check_points/MERT_FF_transfer_learning_epoch_{epoch+1}.pth"
            best_Final_FF_path = f"check_points/Final_FF_transfer_learning_epoch_{epoch+1}.pth"
            torch.save(CQTNet_model.state_dict(), best_CQTNet_path)
            torch.save(MERT_FF.state_dict(), best_MERT_FF_path)
            torch.save(Final_FF.state_dict(), best_Final_FF_path)
            print(f"New best model saved to: {best_CQTNet_path}, {best_MERT_FF_path}, {best_Final_FF_path}")
    
    # Load best model and evaluate on test set
    CQTNet_model.load_state_dict(torch.load(best_CQTNet_path))
    MERT_FF.load_state_dict(torch.load(best_MERT_FF_path))
    Final_FF.load_state_dict(torch.load(best_Final_FF_path))
    test_map, test_top10, test_rank1 = val_slow(CQTNet_model, MERT_FF, Final_FF, test_CQTNet_loader, test_MERT_loader, -1, "Indian Test Set")
    print(f"Final Test Set Performance - MAP: {test_map:.4f}, Top10: {test_top10:.4f}, Rank1: {test_rank1:.2f}")

@torch.no_grad()
def val_slow(CQTNet_model, MERT_FF, Final_FF, CQTNet_loader, MERT_loader, epoch, dataset_name=None):
    CQTNet_model.eval()
    MERT_FF.eval()
    Final_FF.eval()
    all_embeddings = []
    all_labels = []
    
    for (MERT_data, MERT_label), (CQTNet_data, CQTNet_label) in tqdm(zip(MERT_loader, CQTNet_loader), desc=f"Evaluating {dataset_name}"):
        assert MERT_label == CQTNet_label
        
        with torch.no_grad():
            MERT_FF_out = MERT_FF(MERT_data.to(opt.device))
            feat = CQTNet_model(CQTNet_data.to(opt.device))
            _,embedding = Final_FF(feat.to(opt.device), MERT_FF_out.to(opt.device))
        all_embeddings.append(embedding.cpu().numpy())
        all_labels.append(int(CQTNet_label))
    
    embeddings = np.concatenate(all_embeddings)
    embeddings = norm(embeddings)
    
    dis2d = -np.matmul(embeddings, embeddings.T)
    MAP, top10, rank1 = calc_MAP(dis2d, all_labels)
    
    return MAP, top10, rank1

if __name__=='__main__':
    import fire
    fire.Fire()
