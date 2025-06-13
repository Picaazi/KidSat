import argparse
import pandas as pd
from tqdm import tqdm
import os
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import L1Loss
import warnings
from sklearn.model_selection import train_test_split
from preparation import set_seed, save_checkpoint, get_datasets
from sklearn.preprocessing import StandardScaler
from models import  PovertySiren
warnings.filterwarnings("ignore")


class SirenDataset(Dataset):
    """Dataset for Siren coordinate-based training - matches filtered DINOv2 dataset"""
    def __init__(self, dataframe, predict_target, coord_scaler=None):
        self.dataframe = dataframe
        self.predict_target = predict_target
        
        # Extract coordinates
        self.coordinates = np.column_stack([
            dataframe['LATNUM'].values, 
            dataframe['LONGNUM'].values
        ])
        
        # Scale coordinates
        if coord_scaler is None:
            self.coord_scaler = StandardScaler()
            self.coordinates_scaled = self.coord_scaler.fit_transform(self.coordinates)
        else:
            self.coord_scaler = coord_scaler
            self.coordinates_scaled = self.coord_scaler.transform(self.coordinates)
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        
        # Get coordinates as tensor
        coords_tensor = torch.tensor(self.coordinates_scaled[idx], dtype=torch.float32)
        
        # Get target as tensor (99/101-dimensional poverty vector)
        target = torch.tensor(item[self.predict_target], dtype=torch.float32)
        
        return coords_tensor, target

def main(fold, target, imagery_path, imagery_source, representation_dim, hidden_dim, 
         num_layers, batch_size, num_epochs, country=None, enhanced_targets=False):
    
    print(f"Starting Siren training for fold {fold}")
    print(f"Imagery source: {imagery_source} (used for dataset filtering)")
    print(f"Imagery path: {imagery_path}")
    
    data_folder = r'survey_processing/processed_data'
    country_suffix = f'_{country.upper()}' if country else ''
    enhanced_suffix = f'_enhanced' if enhanced_targets else ''
    
    # Load data
    train_df = pd.read_csv(f'{data_folder}/train_fold_{fold}{country_suffix}.csv')
    test_df = pd.read_csv(f'{data_folder}/test_fold_{fold}{country_suffix}.csv')
    
    # Model paths
    best_model = f'modelling/dino/model/siren_spatial_{fold}_best{country_suffix}{enhanced_suffix}.pth'
    last_model = f'modelling/dino/model/siren_spatial_{fold}_last{country_suffix}{enhanced_suffix}.pth'
    
    print(f"Model files:")
    print(f"  Best: {best_model}")
    print(f"  Last: {last_model}")
    
    # Create model directory
    os.makedirs(os.path.dirname(best_model), exist_ok=True)

    # CRITICAL: Use get_datasets to filter exactly like DINOv2
    # This ensures Siren trains on the same locations as DINOv2
    print("Filtering datasets using get_datasets (same as DINOv2)...")
    train_df, test_df, predict_target = get_datasets(train_df, test_df, imagery_path, imagery_source, target, enhanced_targets)
    
    print(f"Enhanced fine-tuning: {enhanced_targets}")
    print(f"Number of target variables: {len(predict_target)}")
    print(f"Target variables: {predict_target}")
    
    # Compare with expected 99 variables
    expected_targets = ['h10', 'h3', 'h31', 'h5', 'h7', 'h9', 
                    'hc70', 'hv109', 'hv121', 'hv106', 'hv201', 
                    'hv204', 'hv205', 'hv216', 'hv225', 'hv271', 'v312']

    if enhanced_targets:
        expected_targets.append('hv025')

    # Check which base variables are missing
    missing_base = [col for col in expected_targets if not any(col in t for t in predict_target)]
    print(f"Missing base variables: {missing_base}")


    # Set seed 
    seed = 42
    set_seed(seed)
    
    # Train/validation split
    train, validation = train_test_split(train_df, test_size=0.2, random_state=seed)
    
    print(f"Final training samples: {len(train)}")
    print(f"Final validation samples: {len(validation)}")

        # Create datasets - now using the same filtered data as DINOv2
    train_dataset = SirenDataset(train, predict_target)
    val_dataset = SirenDataset(validation, predict_target, coord_scaler=train_dataset.coord_scaler)
    
    # Create data loaders with batching (like DINOv2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")


    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Initialize model
    model = PovertySiren(
        input_dim=2,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        representation_dim=representation_dim
    ).to(device)
    
    # Modify the prediction head to match your target dimensions
    model.prediction_head = nn.Sequential(
        nn.Linear(representation_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, len(predict_target)),  # Output size matches target variables
        nn.Sigmoid()
    ).to(device)
    
    # Load existing model if available
    if os.path.exists(last_model):
        last_state_dict = torch.load(last_model)
        best_error = torch.load(best_model)['loss']
        epochs_ran = last_state_dict['epoch']
        model.load_state_dict(last_state_dict['model_state_dict'])
        print('Found existing model')
    else:
        epochs_ran = 0
        best_error = np.inf
    
    # Optimizer setup - following your pattern (same as DINOv2)
    base_model_params = {'params': model.encoder_layers.parameters(), 'lr': 1e-6, 'weight_decay': 1e-6}
    head_params = {'params': model.prediction_head.parameters(), 'lr': 1e-6, 'weight_decay': 1e-6}
    
    optimizer = torch.optim.Adam([base_model_params, head_params])
    loss_fn = L1Loss()

    # Training loop
    for epoch in range(epochs_ran + 1, num_epochs):
        torch.cuda.empty_cache()
        model.train()
        print('Training...')
        
        # Training phase with batching (exactly like DINOv2)
        for batch in tqdm(train_loader):
            coordinates, targets = batch
            coordinates, targets = coordinates.to(device), targets.to(device)
            
            # Forward pass
            poverty_pred, representation = model(coordinates)
            loss = loss_fn(poverty_pred, targets)
            
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
            coordinates, targets = batch
            coordinates, targets = coordinates.to(device), targets.to(device)
            
            # Forward pass
            with torch.no_grad():
                poverty_pred, representation = model(coordinates)
            
            batch_loss = loss_fn(poverty_pred, targets)
            val_loss.append(batch_loss.item())
            indiv_loss.append(torch.mean(torch.abs(poverty_pred - targets), axis=0))
        
        # Compute mean validation loss
        mean_val_loss = np.mean(val_loss)   
        mean_indiv_loss = torch.stack(indiv_loss).mean(dim=0)

        # Save best model - following your pattern
        if mean_val_loss < best_error:
            save_checkpoint(model, optimizer, epoch, mean_val_loss, filename=best_model)
            # Also save the coordinate scaler for later use
            checkpoint = torch.load(best_model)
            checkpoint['coord_scaler'] = train_dataset.coord_scaler
            checkpoint['representation_dim'] = representation_dim
            checkpoint['hidden_dim'] = hidden_dim
            checkpoint['num_layers'] = num_layers
            checkpoint['predict_target'] = predict_target
            torch.save(checkpoint, best_model)
            best_error = mean_val_loss
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {mean_val_loss}, Individual Loss: {mean_indiv_loss}')
        
        # Save last model - following your pattern
        save_checkpoint(model, optimizer, epoch, mean_val_loss, filename=last_model)
        
        # Also save coordinate scaler in last model
        checkpoint = torch.load(last_model)
        checkpoint['coord_scaler'] = train_dataset.coord_scaler
        checkpoint['representation_dim'] = representation_dim
        checkpoint['hidden_dim'] = hidden_dim
        checkpoint['num_layers'] = num_layers
        checkpoint['predict_target'] = predict_target
        torch.save(checkpoint, last_model)

    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_error:.4f}")
    print(f"Model trained on {len(train)} locations (same as DINOv2)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Siren location model for poverty prediction.')
    parser.add_argument('--fold', type=str, help='CV fold')
    parser.add_argument('--target', type=str, default='', help='Target variable (empty for multi-target)')
    parser.add_argument('--imagery_path', type=str, required=True, help='The parent directory of all imagery (for dataset filtering)')
    parser.add_argument('--imagery_source', type=str, default='L', help='L for Landsat and S for Sentinel (for dataset filtering)')
    parser.add_argument('--representation_dim', type=int, default=128, help='Dimension of learned representation')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of hidden layers')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--country', type=str, help='Two-letter country code for single country training (e.g., ET, KE)')
    parser.add_argument('--enhanced_targets', action='store_true', help='Include hv025 in fine-tuning targets')
    
    args = parser.parse_args()

    # Validate country code if provided
    if args.country:
        args.country = args.country.upper()
        if len(args.country) != 2:
            raise ValueError("Country code must be exactly 2 letters (e.g., ET, KE)")
        
    main(args.fold, args.target, args.imagery_path, args.imagery_source, args.representation_dim, 
         args.hidden_dim, args.num_layers, args.batch_size, args.num_epochs, args.country, args.enhanced_targets)