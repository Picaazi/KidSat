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
import torchvision.models as models
from PIL import Image    
import re
import torch.nn as nn
import imageio
from sklearn.model_selection import train_test_split    
from torch.optim import Adam
from torch.nn import L1Loss
import warnings
warnings.filterwarnings("ignore")
import sys
from preparation import set_seed, CustomDataset, save_checkpoint, get_datasets
from models import ViTForRegression
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    
def main(fold, model_name, target, imagery_path, imagery_source, emb_size, batch_size, num_epochs, img_size = None, raw = False):
    
    if imagery_source == 'L':
        normalization = 30000.
        imagery_size = 336
    elif imagery_source == 'S':
        normalization = 3000.
        imagery_size = 1024
    else:
        raise Exception("Unsupported imagery source")
    
    if not img_size is None:
        imagery_size = img_size
        
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
        transforms.Normalize(mean=[0.5] * 13, std=[0.22] * 13),  # Normalize with ImageNet's mean and std
    ])

    train_dataset = CustomDataset(train, transform, normalization, predict_target, all=True)
    val_dataset = CustomDataset(validation, transform, normalization, predict_target, all=True)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size+4)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size+4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_input_channels = 13  # For example, 13 channels

    # Load the Swin-Base model without pretrained weights
    model = models.swin_b(weights=None)
    if not raw:
        model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
    # Modify the first convolution layer to accept a different number of input channels
    # The original layer has an in_channels of 3
    model.features[0][0] = nn.Conv2d(
        in_channels=num_input_channels,
        out_channels=model.features[0][0].out_channels,
        kernel_size=model.features[0][0].kernel_size,
        stride=model.features[0][0].stride,
        padding=model.features[0][0].padding,
        bias=(model.features[0][0].bias is not None)
    )
    base_model = model

    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    model = ViTForRegression(base_model, emb_size=emb_size, predict_size=len(predict_target)).to(device)
    best_model = f'modelling/dino/model/{model_name}_{fold}_all_cluster_best_{imagery_source}{target}_.pth'
    last_model = f'modelling/dino/model/{model_name}_{fold}_all_cluster_last_{imagery_source}{target}_.pth'
    
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

    base_model_params = {'params': model.base_model.parameters(), 'lr': 1e-5, 'weight_decay': 1e-6}
    head_params = {'params': model.regression_head.parameters(), 'lr': 1e-5, 'weight_decay': 1e-6}

    # Setup the optimizer
    optimizer = Adam([base_model_params, head_params])
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
    parser.add_argument('--emb_size', type=int, default=1000, help='Size of the model output')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--imagery_size', type=int, help='Size of the imagery')
    parser.add_argument('--raw', action='store_true', help='Use raw model')
    args = parser.parse_args()
    main(args.fold, args.model_name, args.target, args.imagery_path, args.imagery_source,args.emb_size, args.batch_size, args.num_epochs, args.imagery_size, args.raw)