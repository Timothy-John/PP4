import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import models
from config import opt
from utility import *
from tqdm import tqdm
import numpy as np
from cqt_loader import IndianCoverCQT
import random

#Setting Randomization Seed for Reproducibility
random.seed(7)
torch.manual_seed(7)

def custom_collate(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    data = torch.stack(data)
    labels = torch.LongTensor(labels)
    return data, labels

def transfer_learning(**kwargs):
    opt.batch_size = 32
    opt.num_workers = 2
    opt.model = 'CQTNet'
    opt.load_model_path = '/content/drive/MyDrive/CoverSongDetection_Timothy/CQTNet_SpecAugment_x3.pth'
    # opt.load_model_path = '/content/CQTNet/check_points/latest.pth'
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt._parse(kwargs)
    print(f"Using device: {opt.device}")

    # Load pre-trained model
    model = getattr(models, opt.model)()
    model.load(opt.load_model_path)
    model = model.to(opt.device)

    # Only 1st conv layer frozen
    for name, param in model.named_parameters():
       if 'conv0' in name:
           param.requires_grad = False
       else:
           param.requires_grad = True
    """
    # Freeze all layers except the last two
    for name, param in model.named_parameters():
        if 'fc0' not in name and 'fc1' not in name:
            param.requires_grad = False
    """

    # Modify the last layer to output embeddings
    model.fc1 = nn.Linear(300, 300).to(opt.device)  # Change output to 300-dimensional embedding

    # Prepare Indian Cover Songs dataset
    train_data = IndianCoverCQT('train')
    val_data = IndianCoverCQT('val')
    test_data = IndianCoverCQT('test')

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=custom_collate)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)

    # Define loss function and optimizer
    criterion = nn.TripletMarginLoss(margin=0.3)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Training loop
    num_epochs = 200
    best_val_map = 0
    best_model_path = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(opt.device), labels.to(opt.device)

            optimizer.zero_grad()
            embeddings, _ = model(inputs)
            
            # Create triplets
            anchor, positive, negative = create_triplets(embeddings, labels)
            
            if anchor.size(0) > 0:  # Check if we have valid triplets
                loss = criterion(anchor, positive, negative)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            else:
                print("No valid triplets in this batch. Skipping.")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Evaluate on validation set
        val_map, val_top10, val_rank1 = val_slow(model, val_loader, epoch, "Indian Validation Set")
        print(f"Validation - MAP: {val_map:.4f}, Top10: {val_top10:.4f}, Rank1: {val_rank1:.2f}")

        if val_map > best_val_map:
            best_val_map = val_map
            best_model_path = f"check_points/CQTNet_transfer_learning_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(best_model_path))
    test_map, test_top10, test_rank1 = val_slow(model, test_loader, -1, "Indian Test Set")
    print(f"Final Test Set Performance - MAP: {test_map:.4f}, Top10: {test_top10:.4f}, Rank1: {test_rank1:.2f}")

    return best_model_path

def create_triplets(embeddings, labels):
    """
    Create triplets for triplet loss
    """
    triplets = []
    for i in range(len(embeddings)):
        anchor = embeddings[i].unsqueeze(0)
        positive_indices = (labels == labels[i]).nonzero().squeeze()
        negative_indices = (labels != labels[i]).nonzero().squeeze()
        
        # Handle cases where indices might be 0-d tensors
        if positive_indices.dim() == 0:
            positive_indices = positive_indices.unsqueeze(0)
        if negative_indices.dim() == 0:
            negative_indices = negative_indices.unsqueeze(0)
        
        if len(positive_indices) > 1 and len(negative_indices) > 0:
            positive_index = random.choice(positive_indices.tolist())
            while positive_index == i:
                positive_index = random.choice(positive_indices.tolist())
            negative_index = random.choice(negative_indices.tolist())
            
            positive = embeddings[positive_index].unsqueeze(0)
            negative = embeddings[negative_index].unsqueeze(0)
            
            triplets.append((anchor, positive, negative))
    
    if triplets:
        anchors, positives, negatives = zip(*triplets)
        return torch.cat(anchors).to(opt.device), torch.cat(positives).to(opt.device), torch.cat(negatives).to(opt.device)
    else:
        return embeddings[0].unsqueeze(0), embeddings[0].unsqueeze(0), embeddings[0].unsqueeze(0)

@torch.no_grad()
def val_slow(model, dataloader, epoch, dataset_name=None):
    model.eval()
    all_embeddings = []
    all_labels = []

    for data, label in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
        input = data.to(opt.device)
        embedding, _ = model(input)
        all_embeddings.append(embedding.cpu().numpy())
        all_labels.append(label.cpu().numpy())

    embeddings = np.concatenate(all_embeddings)
    labels = np.concatenate(all_labels)
    
    embeddings = norm(embeddings)
    dis2d = -np.matmul(embeddings, embeddings.T)
    
    MAP, top10, rank1 = calc_MAP(dis2d, labels)

    print(f"\nResults for {dataset_name}:")
    print(f"MAP: {MAP:.4f}, Top10: {top10:.4f}, Rank1: {rank1:.2f}")
    return MAP, top10, rank1

if __name__=='__main__':
    import fire
    fire.Fire()
