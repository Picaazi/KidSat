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
import torchvision.models as models
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preparation import image_config, set_seed, save_checkpoint, CustomDataset, get_datasets
from models import ViTForRegression

def main(fold, model_name, target, imagery_path, imagery_source, emb_size, batch_size, num_epochs, img_size = None,sigmoid = True):
    normalization, imagery_size = image_config(imagery_source, img_size)

    data_folder = r'survey_processing/processed_data'

    train_df = pd.read_csv(f'{data_folder}/train_fold_{fold}.csv')
    test_df = pd.read_csv(f'{data_folder}/test_fold_{fold}.csv')
    
    train_df, test_df, predict_target = get_datasets(train_df, test_df, imagery_path, imagery_source)

    # Set your desired seed
    seed = 42
    set_seed(seed)
    train, validation = train_test_split(train_df, test_size=0.2, random_state=seed)

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Resize((imagery_size, imagery_size)),  # Resize the image to the input size expected by the model
        # transforms.Normalize(mean=[0.5] * 13, std=[0.22] * 13),  # Normalize with ImageNet's mean and std
    ])

    train_dataset = CustomDataset(train, transform, normalization, predict_target, all=True)
    val_dataset = CustomDataset(validation, transform, normalization, predict_target, all=True)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size+4)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size+4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
    
    
    class BandSelector(nn.Module):
        # Used to select specific bands from the input tensor
        # This model takes a 13-channel input and outputs a 3-channel tensor
    
        def __init__(self):
            super().__init__()
            # Define a 1x1 convolution to map 13 channels to 3 channels
            self.conv = nn.Conv2d(13, 3, kernel_size=1, bias=False)
            
            # Initialize all weights to small values
            nn.init.normal_(self.conv.weight, mean=0.0, std=0.01)
            
            # Manually set weights for bands 4, 3, 2 (input channels 3, 2, 1) to 1.0
            self.conv.weight.data[0, 3] = 1.0  # Output channel 0: Band 4 (input channel 3)
            self.conv.weight.data[1, 2] = 1.0  # Output channel 1: Band 3 (input channel 2)
            self.conv.weight.data[2, 1] = 1.0  # Output channel 2: Band 2 (input channel 1)
        # Forward pass through the convolution layer
        def forward(self, x):
            return self.conv(x)
    
    # Move the updated model to the device
    base_model = base_model.to(device)

    torch.cuda.empty_cache()
    projection = BandSelector().to(device)

    print(f"Using {device}")
    act_name = 'sigmoid' if sigmoid else 'clipped_relu'
    model = ViTForRegression(base_model, projection, activation = act_name, emb_size=emb_size, predict_size=len(predict_target)).to(device)
    
    best_model = f'modelling/dino/model/{model_name}_{fold}_all_cluster_best_{imagery_source}{target}_.pth'
    last_model = f'modelling/dino/model/{model_name}_{fold}_all_cluster_last_{imagery_source}{target}_.pth'
    
    # Check if the model already exists
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

    base_model_params = {'params': model.base_model.parameters(), 'lr': 1e-6, 'weight_decay': 1e-6}
    head_params = {'params': model.regression_head.parameters(), 'lr': 1e-6, 'weight_decay': 1e-6}

    # Setup the optimizer
    optimizer = torch.optim.Adam([base_model_params, head_params])
    loss_fn = L1Loss()

    for epoch in range(epochs_ran+1, num_epochs):
        torch.cuda.empty_cache()
        model.train()
        print('Training...')
        for batch in tqdm(train_loader):
            images, targets = batch
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            
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
                outputs = model(images)
            batch_loss = loss_fn(outputs, targets)
            val_loss.append(batch_loss.item())
            indiv_loss.append(torch.mean(torch.abs(outputs-targets), axis=0))
        
        # Compute mean validation loss
        mean_val_loss = np.mean(val_loss)   
        mean_indiv_loss = torch.stack(indiv_loss).mean(dim=0)

        # If we have a better model, save it
        if mean_val_loss< best_error:
            save_checkpoint(model, optimizer, epoch, mean_val_loss, filename=best_model)
            best_error = mean_val_loss
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {mean_val_loss}, Individual Loss: {mean_indiv_loss}')
        save_checkpoint(model, optimizer, epoch, mean_val_loss, filename=last_model)


# Command line
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