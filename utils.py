import os
import glob
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from loaders import get_loader_model

from cqt_loader import IndianCoverCQT
from cqtnet_utility import *
from tqdm import tqdm
from torch import nn
import models


WINDOW_SECONDS = 2
STEP_SECONDS = 1

def shuffle(a, b):
    c = list(zip(a, b))
    random.shuffle(c)
    return zip(*c)

def custom_collate(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    data = torch.Tensor(data)
    labels = torch.LongTensor(labels)
    return data.to("cuda"), labels.to("cuda")


def load_dataset(model_name, win_seconds=WINDOW_SECONDS, step_seconds=STEP_SECONDS):
    dataset_base_folder = f'../CoverSongDetection_Timothy/Encodec/dataset/{model_name}'
    indir = '../CoverSongDetection_Timothy/CoverIndian_audio'
    filepath = 'data/coversIndian_list.txt'
    with open(filepath, 'r') as fp:
        file_list = [line.rstrip() for line in fp]
    
    loader_model = get_loader_model(model_name, win_seconds, step_seconds)
    if not os.path.exists(dataset_base_folder):
        os.mkdir(dataset_base_folder)
        for filename in file_list:
            in_path = indir+"/"+filename[:-int(int(len(filename.split('_')[-1]))+1)] +'/' +filename +'.mp3'
            X = loader_model.load(in_path)
            if len(X) == 0:
                print(f"No files found")
                continue
            else:
                X = np.array(X)
                np.save(dataset_base_folder+"/"+filename+".npy", X)


def get_MNIST_train_model(classes, dataset_name, model_name, channels=256, feature_len=128, kernel_size=5, max_pool_size=5):
    model = torch.nn.Sequential(
        torch.nn.Conv2d(channels, feature_len, kernel_size),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(max_pool_size),
        torch.nn.Conv2d(feature_len, feature_len*2, kernel_size),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(max_pool_size),
        torch.nn.Dropout(),
        torch.nn.Flatten(),
        torch.nn.LazyLinear(300),
        torch.nn.LazyLinear(classes)
    )
    return model.to("cuda")
    

def check_step(loader, classification_model, epoch):
    classification_model.eval()
    all_embeddings = []
    all_labels = []
    for inputs, label in loader:
        embedding = classification_model[:9](inputs.to("cuda"))
        all_embeddings.append(embedding.cpu().numpy())
        all_labels.append(label.cpu().numpy())
    embeddings = np.concatenate(all_embeddings)
    labels = np.concatenate(all_labels)
    embeddings = norm(embeddings)
    dis2d = -np.matmul(embeddings, embeddings.T)
    return calc_MAP(dis2d, labels)

def train_loop(classification_model, model_name, optimizer=torch.optim.SGD, lr=0.01, criterion=torch.nn.CrossEntropyLoss, epochs=20, batch_size=32):
    criterion = criterion()
    optimizer = optimizer(classification_model.parameters(), lr=lr)
    train_data = IndianCoverCQT(model_name, 'train')
    val_data = IndianCoverCQT(model_name, 'val')
    test_data = IndianCoverCQT(model_name, 'test')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=custom_collate)
    all_train_loss = []
    all_val_map = []
    all_train_acc = []
    best_val_map = -float("inf")
    best_model = None
    train_val_filepath = 'data/coversIndian_train_val.txt'
    with open(train_val_filepath, 'r') as fp:
        train_val_file_list = [line.rstrip() for line in fp]
    test_filepath = 'data/coversIndian_test.txt'
    with open(test_filepath, 'r') as fp:
        test_file_list = [line.rstrip() for line in fp]
    # Perform training
    for epoch in tqdm(range(epochs)):
        # Iterate over train set
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = classification_model(inputs.to("cuda"))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # Compute train metrics
        t_loss = loss.item()
        all_train_loss.append(t_loss)
        # Compute validation accuracy and loss and save best model
        with torch.no_grad():
            val_map, val_top10, val_rank1 = check_step(val_loader, classification_model, epoch)
        # Compute validation metrics
        if val_map > best_val_map:
            best_val_map = val_map
            best_val_top10 = val_top10
            best_val_rank1 = val_rank1
            best_model = classification_model.state_dict()
        all_val_map.append(val_map)
        if epoch % 10 == 0:
            print(f'{epoch}/{epochs} - Train Loss: {t_loss}. Val MAP: {val_map}. Val Top10: {val_top10}. Val Rank1: {val_rank1}')
    print(f'\nBEST MODEL:: Val MAP: {best_val_map}. Val Top10: {best_val_top10}. Val Rank1: {best_val_rank1}\n')
    classification_model.load_state_dict(best_model)
    with torch.no_grad():
        test_map, test_top10, test_rank1 = check_step(test_loader, classification_model, -1)

    print(f'Test MAP: {test_map}. Test Top10: {test_top10}. Test Rank1: {test_rank1}')
    return all_train_loss, best_val_map, best_val_top10, best_val_rank1, test_map, test_top10, test_rank1, best_model


def perform_training(dataset_name, model_name, results_folder="results"):
    load_dataset(model_name)
    #model = get_MNIST_train_model(300, dataset_name, model_name)
    model = getattr(models, 'CQTNet')()
    model.load('../CoverSongDetection_Timothy/CQTNet_SpecAugment_x3.pth')
    model.fc1 = nn.Linear(300, 300)
    model = model.to("cuda")
    
    train_loss, best_val_map, best_val_top10, best_val_rank1, test_map, test_top10, test_rank1, best_model = train_loop(model, model_name)
    torch.save(best_model, f"Transfer_Learning_classify_{dataset_name}_{model_name}_model.pth")
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    with open(f"{results_folder}/classify_{dataset_name}_{model_name}.json", "w") as f:
        json.dump(
            {
                'train_loss': train_loss,
                'best_val_MAP': best_val_map,
                'best_val_Top10': best_val_top10,
                'best_val_Rank1': best_val_rank1,
                'test_MAP': test_map,
                'test_Top10': test_top10,
                'test_Rank1': test_rank1
            }, f)
