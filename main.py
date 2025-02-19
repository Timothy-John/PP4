import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import models
from config import opt

#from utility import *
from sklearn.metrics import average_precision_score

from tqdm import tqdm
import numpy as np
from cqt_loader import IndianCoverCQT
import random


def custom_collate(batch):
    data1 = [item[0][0] for item in batch]
    data2 = [item[0][1] for item in batch]
    labels = [item[1] for item in batch]
    data1 = torch.stack(data1)
    data2 = torch.stack(data2)
    labels = torch.Tensor(labels)
    return (data1, data2), labels

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
    model.load_state_dict(torch.load(opt.load_model_path), strict=False)
    model = model.to(opt.device)

    #model.fc1 = nn.Linear(300, 100).to(opt.device)
    #model = torch.nn.Sequential(model, nn.Linear(100,1)).to(opt.device)

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

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=custom_collate)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)

    # Define loss function and optimizer
    criterion = torch.nn.BCELoss()  #WithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Training loop
    num_epochs = 10
    best_val_map = 0
    best_model_path = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for (inputs1, inputs2), labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs1, inputs2, labels = inputs1.to(opt.device), inputs2.to(opt.device), labels.to(opt.device)

            optimizer.zero_grad()
            scores = model([inputs1, inputs2])
            scores = scores.squeeze()

            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Evaluate on validation set
        val_map = val_slow(model, val_loader, epoch, "Indian Validation Set")
        print(f"Validation - MAP: {val_map:.4f}")

        if val_map > best_val_map:
            best_val_map = val_map
            best_model_path = f"check_points/CQTNet_transfer_learning_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(best_model_path))
    test_map = val_slow(model, test_loader, -1, "Indian Test Set")
    print(f"Final Test Set Performance - MAP: {test_map:.4f}")

    return best_model_path

@torch.no_grad()
def val_slow(model, dataloader, epoch, dataset_name=None):
    model.eval()
    all_scores = []
    all_labels = []

    for (data1, data2), label in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
        input1, input2 = data1.to(opt.device), data2.to(opt.device)
        score = model([input1, input2])
        score, label = score.squeeze().item(), label.item()
        all_scores.append(score)
        all_labels.append(label)
    MAP = average_precision_score(np.array(all_labels), np.array(all_scores))

    print(f"\nResults for {dataset_name}:")
    print(f"MAP: {MAP:.4f}")
    return MAP

if __name__=='__main__':
    import fire
    fire.Fire()
