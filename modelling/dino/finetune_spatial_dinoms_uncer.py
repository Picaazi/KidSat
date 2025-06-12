import argparse
import pandas as pd
from tqdm import tqdm
import os
import random
import rasterio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from PIL import Image    
import re
import torch.nn as nn
import imageio
from sklearn.model_selection import train_test_split    
from torch.optim import Adam
from torch.nn import L1Loss
import warnings
from preparation import image_config, set_seed, CustomDataset, get_datasets
from models import ViTForRegressionWithUncertainty
warnings.filterwarnings("ignore")

def main(fold, model_name, target, imagery_path, imagery_source, emb_size, batch_size, num_epochs, img_size = None):
    
    normalization, imagery_size = image_config(imagery_source, img_size)
        
    data_folder = r'survey_processing/processed_data'
    
    train_df = pd.read_csv(f'{data_folder}/train_fold_{fold}.csv')
    test_df = pd.read_csv(f'{data_folder}/test_fold_{fold}.csv')
    
    train_df, test_df, predict_target = get_datasets(train_df, test_df, imagery_path, imagery_source, target)

    # Set your desired seed
    seed = 42
    set_seed(seed)
    train, validation = train_test_split(train_df, test_size=0.2, random_state=seed)

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Resize((imagery_size, imagery_size)),  # Resize the image to the input size expected by the model
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet's mean and std
    ])

    train_dataset = CustomDataset(train, transform, normalization, predict_target, all=True)
    val_dataset = CustomDataset(validation, transform, normalization, predict_target, all=True)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size+4)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size+4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    base_models = [torch.hub.load('facebookresearch/dinov2', model_name).to(device) for _ in range(3)]

    def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth"):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, filename)

    torch.cuda.empty_cache()

    model = ViTForRegressionWithUncertainty(base_models, emb_size=emb_size).to(device)
    best_model = f'modelling/dino/model/{model_name}ms_uncer_{fold}_all_cluster_best_{imagery_source}{target}_.pth'
    last_model = f'modelling/dino/model/{model_name}ms_uncer_{fold}_all_cluster_last_{imagery_source}{target}_.pth'
    if os.path.exists(last_model):
        last_state_dict = torch.load(last_model)
        best_error = torch.load(best_model)['loss']
        epochs_ran = last_state_dict['epoch']
        model.load_state_dict(last_state_dict['model_state_dict'])
        print('Found existing model')
    else:
        epochs_ran = 0
        best_error = np.inf

    # Move model to appropriate device
    model.to(device)

    base_model_params = [{'params': model.parameters(), 'lr': 1e-6, 'weight_decay': 1e-6} for model in model.base_models]
    head_params = {'params': model.regression_head.parameters(), 'lr': 1e-6, 'weight_decay': 1e-6}

    # Setup the optimizer
    optimizer = torch.optim.Adam(base_model_params+ [head_params])

    def nll_loss(mean, variance, targets):
        # Avoid negative variance by ensuring a small positive value
        eps = 1e-6
        variance = variance + eps

        # Calculate the negative log-likelihood
        loss = 0.5 * ((mean - targets) ** 2) / variance + 0.5 * torch.log(variance)
        return torch.mean(loss)
    
    loss_fn = nll_loss
    for epoch in range(epochs_ran+1, num_epochs):
        torch.cuda.empty_cache()
        model.train()
        print('Training...')
        for batch in tqdm(train_loader):
            images, targets = batch
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            mean, variance = model(images)
        
            # Calculate the negative log-likelihood loss
            loss = nll_loss(mean, variance, targets)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.empty_cache()
        # Validation phase
        model.eval()
        val_loss = []
        indiv_loss = []
        print('Validating...')
        for batch in val_loader:
            images, targets = batch
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            with torch.no_grad():
                mean, variance = model(images)
            
            batch_loss = nll_loss(mean, variance, targets)
            val_loss.append(batch_loss.item())
            indiv_loss.append(torch.mean(torch.abs(mean-targets), axis=0))
        
        # Compute mean validation loss
        mean_val_loss = np.mean(val_loss)   
        mean_indiv_loss = torch.stack(indiv_loss).mean(dim=0)

        if mean_val_loss< best_error:
            save_checkpoint(model, optimizer, epoch, mean_val_loss, filename=best_model)
            best_error = mean_val_loss
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {mean_val_loss}, Individual Loss: {mean_indiv_loss}')
        save_checkpoint(model, optimizer, epoch, mean_val_loss, filename=last_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run satellite image processing model training.')
    parser.add_argument('--fold', type=str, help='CV fold')
    parser.add_argument('--model_name', type=str, help='Name of the model')
    parser.add_argument('--target', type=str,default='', help='Target variable')
    parser.add_argument('--imagery_path', type=str, help='The parent directory of all imagery')
    parser.add_argument('--imagery_source', type=str, default='L', help='L for Landsat and S for Sentinel')
    parser.add_argument('--emb_size', type=int, default=768, help='Size of the model output')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--imagery_size', type=int, help='Size of the imagery')
    args = parser.parse_args()
    main(args.fold, args.model_name, args.target, args.imagery_path, args.imagery_source,args.emb_size, args.batch_size, args.num_epochs, args.imagery_size)