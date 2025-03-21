import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import models
from config import opt
from utility import *
from tqdm import tqdm
import numpy as np
from cqt_loader import IndianCoverCQT
import random

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

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
    opt.load_model_path = '../CoverSongDetection_Timothy/CQTNet_SpecAugment_x3.pth'
    # opt.load_model_path = '/content/CQTNet/check_points/latest.pth'
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt._parse(kwargs)
    print(f"Using device: {opt.device}")

    # Load pre-trained model
    model = getattr(models, opt.model)()
    model.load(opt.load_model_path)
    model = model.to(opt.device)

    # Modify the last layer to no:of labels in our dataset
    model.fc1 = nn.Linear(300, 300).to(opt.device)

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

    # Prepare Indian Cover Songs dataset
    train_data = IndianCoverCQT('train')
    val_data = IndianCoverCQT('val')
    test_data = IndianCoverCQT('test')

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=custom_collate)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Training loop
    num_epochs = 200
    best_val_map = 0
    best_model_path = None
    early_stop = 0
    early_stop_patience = 15

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(opt.device), labels.to(opt.device)

            optimizer.zero_grad()
            scores, _ = model(inputs)
            
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

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
            early_stop = 0
        else:
            early_stop+=1
            if early_stop == early_stop_patience:
                break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(best_model_path))
    test_map, test_top10, test_rank1 = val_slow(model, test_loader, -1, "Indian Test Set")
    print(f"Final Test Set Performance - MAP: {test_map:.4f}, Top10: {test_top10:.4f}, Rank1: {test_rank1:.2f}")
    
    if kwargs.get("fine_tune")==True:
        print("\n\nFine Tuning Model with Triplet Loss....")
        augmented_triplet(model, optimizer, train_data, val_loader, test_loader, opt)
        #fine_tune_model(model, optimizer, train_loader, val_loader, test_loader, opt)

def fine_tune_model(model, optimizer, train_loader, val_loader, test_loader, opt):
    num_epochs = 50
    best_val_map = 0
    best_model_path = None
    early_stop = 0
    early_stop_patience = 15

    #Convert to Embedding Layer
    #model.fc1 = nn.Linear(300, 300).to(opt.device)
    
    for name, param in model.named_parameters():
        if 'conv0' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    criterion = nn.TripletMarginLoss(margin=0.05)
    data = IndianCoverCQT('train')
    train_loader = DataLoader(data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=custom_collate)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(opt.device), labels.to(opt.device)
            _, embeddings = model(inputs)
            # Create triplets
            anchor, positive, negative = create_triplets(embeddings, labels)
            if anchor.size(0) > 0:  # Check if we have valid triplets
                loss = criterion(anchor, positive, negative)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
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
            early_stop = 0
        else:
            early_stop+=1
            if early_stop == early_stop_patience:
                break
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(best_model_path))
    test_map, test_top10, test_rank1 = val_slow(model, test_loader, -1, "Indian Test Set")
    print(f"Final Test Set Performance - MAP: {test_map:.4f}, Top10: {test_top10:.4f}, Rank1: {test_rank1:.2f}")

def augmented_triplet(model, optimizer, train_loader, val_loader, test_loader, opt):
    num_epochs = 50
    best_val_map = 0
    best_model_path = None
    for name, param in model.named_parameters():
        if 'conv0' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    criterion = nn.TripletMarginLoss(margin=0.05)
    for epoch in range(num_epochs):
        model.train()
        pos = []
        for f in train_loader.file_list:
            set_id = f.split('_')[0]
            in_path = os.path.join(train_loader.indir, f + '.npy')
            data = np.load(in_path)
            transform_test = transforms.Compose([
                lambda x: x.astype(np.float32) / (np.max(np.abs(x)) + 1e-6),
                lambda x: torch.Tensor(x),
                lambda x: x.unsqueeze(0),  # Add channel dimension
            ])
            data = transform_test(data)
            data = pad_or_truncate(data, 400, 84)
            data = data.unsqueeze(0).to(opt.device)
            if f.split('_')[-1]=="Original":
                _,anchor = model(data)
                all_embeddings = []
                best_neg_dist = np.inf
                data = IndianCoverCQT('train')
                loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=opt.num_workers, collate_fn=custom_collate)
                for inp, label in loader:
                    with torch.no_grad():
                        _,embedding = model(inp.to(opt.device))
                    all_embeddings.append(embedding[0].cpu())
                all_embeddings = torch.stack(all_embeddings).to(opt.device)
                for i in range(len(all_embeddings)):
                    neg_dist = torch.norm(anchor - all_embeddings[i], dim=1)
                    if neg_dist < best_neg_dist:
                        best_neg_dist = neg_dist
                        negative = all_embeddings[i]
                for positive in pos:
                    loss = criterion(anchor, positive, negative)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                pos=[]
            else:
                _,p = model(data)
                pos.append(p)
        val_map, val_top10, val_rank1 = val_slow(model, val_loader, epoch, "Indian Validation Set")
        print(f"Validation - MAP: {val_map:.4f}, Top10: {val_top10:.4f}, Rank1: {val_rank1:.2f}")
        if val_map > best_val_map:
            best_val_map = val_map
            best_model_path = f"check_points/CQTNet_transfer_learning_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
        test_map, test_top10, test_rank1 = val_slow(model, test_loader, -1, "Indian Test Set")
        print(f"Final Test Set Performance - MAP: {test_map:.4f}, Top10: {test_top10:.4f}, Rank1: {test_rank1:.2f}")

# Including necessary methods
def pad_or_truncate(data, target_length, target_freq=84):
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

def create_triplets(embeddings, labels):
    """
    Create triplets for triplet loss.
    For each anchor, select:
      - a positive (same label) with the maximum distance (hard positive)
      - a negative (different label) with the minimum distance (hard negative)
    """
    triplets = []
    for i in range(len(embeddings)):
        anchor = embeddings[i].unsqueeze(0)
        pos_indices = (labels == labels[i]).nonzero().squeeze()
        neg_indices = (labels != labels[i]).nonzero().squeeze()
        
        # Ensure indices are at least 1D tensors
        if pos_indices.dim() == 0:
            pos_indices = pos_indices.unsqueeze(0)
        if neg_indices.dim() == 0:
            neg_indices = neg_indices.unsqueeze(0)
        
        # Exclude the anchor itself from the positive indices.
        pos_indices = pos_indices[pos_indices != i]
        
        if len(pos_indices) > 0 and len(neg_indices) > 0:
            # Get candidate embeddings from the full dataset.
            pos_candidates = embeddings[pos_indices]
            neg_candidates = embeddings[neg_indices]
            
            # Compute distances between the anchor and all positive candidates.
            # Using Euclidean distance (you may also consider squared distances).
            pos_dists = torch.norm(anchor - pos_candidates, dim=1)
            # Hard positive: the one with the maximum distance.
            pos_idx = torch.argmax(pos_dists).item()
            positive = embeddings[pos_indices[pos_idx]].unsqueeze(0)
            
            # Compute distances between the anchor and all negative candidates.
            neg_dists = torch.norm(anchor - neg_candidates, dim=1)
            # Hard negative: the one with the minimum distance.
            neg_idx = torch.argmin(neg_dists).item()
            negative = embeddings[neg_indices[neg_idx]].unsqueeze(0)
            
            triplets.append((anchor, positive, negative))
    
    if triplets:
        anchors, positives, negatives = zip(*triplets)
        return (torch.cat(anchors).to(opt.device), 
                torch.cat(positives).to(opt.device), 
                torch.cat(negatives).to(opt.device))
    else:
        return embeddings[0].unsqueeze(0), embeddings[0].unsqueeze(0), embeddings[0].unsqueeze(0)


@torch.no_grad()
def val_slow(model, dataloader, epoch, dataset_name=None):
    model.eval()
    all_embeddings = []
    all_labels = []

    for data, label in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
        input = data.to(opt.device)
        _, embedding = model(input)
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
