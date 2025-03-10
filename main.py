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
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    labels = torch.LongTensor(labels)
    return data, labels
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

    # Load pre-trained model
    MERT_model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True, device_map=opt.device)
    MERT_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True, device_map=opt.device)

    opt.model = 'CQTNet'
    opt.load_model_path = '../CoverSongDetection_Timothy/CQTNet_SpecAugment_x3.pth'
    CQTNet_model = getattr(models, 'CQTNet')()
    CQTNet_model.load(opt.load_model_path)
    CQTNet_model.fc1 = nn.Linear(300, 300)
    CQTNet_model = CQTNet_model.to(opt.device)

    MERT_FF = getattr(models, 'MERT_FF')()

    for name, param in CQTNet_model.named_parameters():
        param.requires_grad = True
    for name, param in MERT_FF.named_parameters():
        param.requires_grad = True
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, CQTNet_model.parameters()), lr=1e-4)

    # Training loop
    opt.max_epoch = 50
    best_val_map = 0
    best_CQTNet_path = None
    
    optimizer.zero_grad()
    for epoch in range(opt.max_epoch):
        CQTNet_model.train()
        MERT_model.eval()
        total_loss = 0
        
        for i, ((MERT_data, MERT_labels), (CQTNet_data, CQTNet_labels)) in enumerate(tqdm(zip(train_MERT_loader,train_CQTNet_loader), desc=f"Epoch {epoch+1}/{opt.max_epoch}")):
            assert MERT_labels == CQTNet_labels
            
            # make sure the sample_rate aligned
            MERT_inputs = MERT_processor(MERT_data, sampling_rate=24000, return_tensors="pt", padding=True).to(opt.device)
            
            with torch.no_grad():
                outputs = MERT_model(**MERT_inputs, output_hidden_states=False)
            embeddings = outputs.last_hidden_state.squeeze().mean(-2)

            MERT_FF_out = MERT_FF(embeddings)
            
            labels = torch.LongTensor(CQTNet_labels).to(opt.device)

            optimizer.zero_grad()
            scores, _ = CQTNet_model(CQTNet_data, MERT_FF_out)
            loss = criterion(scores, labels)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_CQTNet_loader)
        print(f"Epoch {epoch+1}/{opt.max_epoch}, Loss: {avg_loss:.4f}")

        # Evaluate on validation set
        val_map, val_top10, val_rank1 = val_slow(CQTNet_model, MERT_model, MERT_processor, MERT_FF, val_CQTNet_loader, val_MERT_loader, epoch, "Indian Validation Set")
        print(f"Validation - MAP: {val_map:.4f}, Top10: {val_top10:.4f}, Rank1: {val_rank1:.2f}")

        if val_map > best_val_map:
            best_val_map = val_map
            best_CQTNet_path = f"check_points/CQTNet_transfer_learning_epoch_{epoch+1}.pth"
            torch.save(CQTNet_model.state_dict(), best_CQTNet_path)
            print(f"New best model saved to {best_CQTNet_path}")
    
    # Load best model and evaluate on test set
    CQTNet_model.load_state_dict(torch.load(best_CQTNet_path))
    MERT_model.load_state_dict(torch.load(best_MERT_path))
    test_map, test_top10, test_rank1 = val_slow(CQTNet_model, MERT_model, MERT_processor, MERT_FF, val_CQTNet_loader, val_MERT_loader, -1, "Indian Test Set")
    print(f"Final Test Set Performance - MAP: {test_map:.4f}, Top10: {test_top10:.4f}, Rank1: {test_rank1:.2f}")

@torch.no_grad()
def val_slow(CQTNet_model, MERT_model, MERT_processor, MERT_FF, CQTNet_loader, MERT_loader, epoch, dataset_name=None):
    CQTNet_model.eval()
    MERT_model.eval()
    all_embeddings = []
    all_labels = []
    
    for (MERT_data, MERT_label), (CQTNet_data, CQTNet_label) in tqdm(zip(MERT_loader, CQTNet_loader), desc=f"Evaluating {dataset_name}"):
        assert MERT_label == CQTNet_label
        MERT_inputs = MERT_processor(MERT_data, sampling_rate=24000, return_tensors="pt")
        
        MERT_inputs = MERT_inputs.to(opt.device)
        with torch.no_grad():
          outputs = MERT_model(**MERT_inputs, output_hidden_states=False)
          embeddings = outputs.last_hidden_state.squeeze().mean(-2)

        MERT_FF_out = MERT_FF(embeddings)
        
        with torch.no_grad():
            _, embedding = CQTNet_model(data, MERT_FF_out)
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
