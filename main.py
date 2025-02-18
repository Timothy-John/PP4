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

from torch.amp import autocast, GradScaler
scaler = GradScaler()


def custom_collate(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return data, labels

def transfer_learning(**kwargs):
    opt._parse(kwargs)
    opt.batch_size = 1
    opt.num_workers = 2
    opt.model = 'MERT'
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {opt.device}")
    
    train_data = IndianCover('train')
    val_data = IndianCover('val')
    test_data = IndianCover('test')

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=custom_collate)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)

    model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True, device_map=opt.device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True, device_map=opt.device)
    
    #print(model)

    for name, param in model.named_parameters():
       if '0' in name or '1' in name:
           param.requires_grad = False
       else:
           param.requires_grad = True
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    # Training loop
    opt.max_epoch = 100
    best_val_map = 0
    best_model_path = None
    
    optimizer.zero_grad()
    for epoch in range(opt.max_epoch):
        model.train()
        total_loss = 0
        iters_to_accumulate = 10
        
        for i, (data, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.max_epoch}")):
            # make sure the sample_rate aligned
            inputs = processor(data, sampling_rate=24000, return_tensors="pt", padding=True).to(opt.device)
            labels = torch.LongTensor(labels).to(opt.device)

            with autocast('cuda'):
                outputs = model(**inputs, output_hidden_states=False)
                loss = criterion(outputs.last_hidden_state.mean(-2), labels)
                loss = loss / iters_to_accumulate
            scaler.scale(loss).backward()
            total_loss += loss.item()
            
            if (i + 1) % iters_to_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{opt.max_epoch}, Loss: {avg_loss:.4f}")

        # Evaluate on validation set
        val_map, val_top10, val_rank1 = val_slow(model, processor, val_loader, epoch, "Indian Validation Set")
        print(f"Validation - MAP: {val_map:.4f}, Top10: {val_top10:.4f}, Rank1: {val_rank1:.2f}")

        if val_map > best_val_map:
            best_val_map = val_map
            best_model_path = f"check_points/MERT_transfer_learning_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(best_model_path))
    test_map, test_top10, test_rank1 = val_slow(model, processor, test_loader, -1, "Indian Test Set")
    print(f"Final Test Set Performance - MAP: {test_map:.4f}, Top10: {test_top10:.4f}, Rank1: {test_rank1:.2f}")

@torch.no_grad()
def val_slow(model, processor, dataloader, epoch, dataset_name=None):
    model.eval()
    all_embeddings = []
    all_labels = []
    
    for data, label in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
        inputs = processor(data, sampling_rate=24000, return_tensors="pt")

        inputs = inputs.to(opt.device)
        with torch.no_grad():
          outputs = model(**inputs, output_hidden_states=True)
        
        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
        time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
        embeddings = time_reduced_hidden_states[11]  #Taking Embeddings from 12th Layer
        
        all_embeddings.append(np.expand_dims(embeddings.cpu().numpy(), axis=0))
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
