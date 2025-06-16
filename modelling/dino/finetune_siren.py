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
from models import  PovertySirenSH
warnings.filterwarnings("ignore")

class SirenSHDataset(Dataset):
    """Dataset for SH + Siren coordinate-based training - matches filtered DINOv2 dataset"""
    def __init__(self, dataframe, predict_target):
        self.dataframe = dataframe
        self.predict_target = predict_target
        
        # Extract coordinates - NO SCALING for SH (coordinates are used raw)
        self.coordinates = np.column_stack([
            dataframe['LATNUM'].values,  # Latitude
            dataframe['LONGNUM'].values  # Longitude
        ])
        
        print(f"SirenSHDataset: {len(self.coordinates)} samples")
        print(f"  Coordinate ranges: Lat [{self.coordinates[:, 0].min():.2f}, {self.coordinates[:, 0].max():.2f}]")
        print(f"                    Lon [{self.coordinates[:, 1].min():.2f}, {self.coordinates[:, 1].max():.2f}]")
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        
        # Get coordinates as tensor (raw coordinates for SH encoding)
        coords_tensor = torch.tensor(self.coordinates[idx], dtype=torch.float32)
        
        # Get target as tensor (99/101-dimensional poverty vector)
        target = torch.tensor(item[self.predict_target], dtype=torch.float32)
        
        return coords_tensor, target
    

def main(fold, target, imagery_path, imagery_source, representation_dim, hidden_dim, 
         num_layers, batch_size, num_epochs, country=None, enhanced_targets=False, sh_L=15):
    
    print(f"Starting SH + Siren training for fold {fold}")
    print(f"Imagery source: {imagery_source} (used for dataset filtering)")
    print(f"Imagery path: {imagery_path}")
    print(f"Spherical Harmonics L: {sh_L}")
    
    data_folder = r'survey_processing/processed_data'
    country_suffix = f'_{country.upper()}' if country else ''
    enhanced_suffix = f'_enhanced' if enhanced_targets else ''
    
    # Load data
    train_df = pd.read_csv(f'{data_folder}/train_fold_{fold}{country_suffix}.csv')
    test_df = pd.read_csv(f'{data_folder}/test_fold_{fold}{country_suffix}.csv')
    
    # Model paths - using sh_siren prefix to distinguish from regular siren
    best_model = f'modelling/dino/model/sh_siren_spatial_{fold}_best{country_suffix}{enhanced_suffix}.pth'
    last_model = f'modelling/dino/model/sh_siren_spatial_{fold}_last{country_suffix}{enhanced_suffix}.pth'
    
    print(f"Model files:")
    print(f"  Best: {best_model}")
    print(f"  Last: {last_model}")
    
    # Create model directory
    os.makedirs(os.path.dirname(best_model), exist_ok=True)

    # CRITICAL: Use get_datasets to filter exactly like DINOv2
    # This ensures SH + Siren trains on the same locations as DINOv2
    print("Filtering datasets using get_datasets (same as DINOv2)...")
    train_df, test_df, predict_target = get_datasets(train_df, test_df, imagery_path, imagery_source, target, enhanced_targets)
    
    print(f"Enhanced fine-tuning: {enhanced_targets}")
    print(f"Number of target variables: {len(predict_target)}")
    print(f"Target variables: {predict_target[:5]}...")  # Show first 5
    
    # Compare with expected 99 variables
    expected_targets = ['h10', 'h3', 'h31', 'h5', 'h7', 'h9', 
                    'hc70', 'hv109', 'hv121', 'hv106', 'hv201', 
                    'hv204', 'hv205', 'hv216', 'hv225', 'hv271', 'v312']

    if enhanced_targets:
        expected_targets.append('hv025')

    # Check which base variables are missing
    missing_base = [col for col in expected_targets if not any(col in t for t in predict_target)]
    if missing_base:
        print(f"Missing base variables: {missing_base}")

    # Set seed 
    seed = 42
    set_seed(seed)
    
    # Train/validation split
    train, validation = train_test_split(train_df, test_size=0.2, random_state=seed)
    
    print(f"Final training samples: {len(train)}")
    print(f"Final validation samples: {len(validation)}")

    # Create datasets - now using SH + Siren dataset (no coordinate scaling)
    train_dataset = SirenSHDataset(train, predict_target)
    val_dataset = SirenSHDataset(validation, predict_target)
    
    # Create data loaders with batching (like DINOv2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Initialize SH + SIREN model
    print("Initializing SH + SIREN model...")
    model = PovertySirenSH(
        sh_L=sh_L,                          # Spherical harmonics degree
        hidden_dim=hidden_dim,              # SIREN hidden dimension
        num_layers=num_layers,              # Number of SIREN layers
        representation_dim=representation_dim # Output representation dimension
    ).to(device)
    
    # Modify the prediction head to match your target dimensions
    # This replaces the default prediction head with one matching your poverty vector size
    model.prediction_head = nn.Sequential(
        nn.Linear(representation_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, len(predict_target)),  # Output size matches target variables
        nn.Sigmoid()
    ).to(device)
    
    print(f"Model architecture:")
    print(f"  SH dimension: {model.sh_encoder.output_dim}")
    print(f"  SIREN hidden: {hidden_dim}")
    print(f"  Representation: {representation_dim}")
    print(f"  Output targets: {len(predict_target)}")
    
    # Load existing model if available
    if os.path.exists(last_model):
        print('Loading existing model...')
        last_state_dict = torch.load(last_model, map_location=device)
        best_checkpoint = torch.load(best_model, map_location=device)
        best_error = best_checkpoint['loss']
        epochs_ran = last_state_dict['epoch']
        model.load_state_dict(last_state_dict['model_state_dict'])
        print(f'Loaded model from epoch {epochs_ran}, best error: {best_error:.4f}')
    else:
        epochs_ran = 0
        best_error = np.inf
        print('Starting fresh training...')
    
    # Optimizer setup - following your pattern but adjusted for SH + SIREN
    # The SH encoder doesn't need training (it's fixed), so we only train SIREN layers
    siren_params = list(model.encoder_layers.parameters()) + list(model.representation_layer.parameters())
    head_params = list(model.prediction_head.parameters())
    
    optimizer = torch.optim.Adam([
        {'params': siren_params, 'lr': 1e-4, 'weight_decay': 1e-6},      # SIREN layers
        {'params': head_params, 'lr': 1e-4, 'weight_decay': 1e-6}       # Prediction head
    ])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    loss_fn = L1Loss()

    print(f"Starting training from epoch {epochs_ran + 1} to {num_epochs}")
    
    # Training loop
    for epoch in range(epochs_ran + 1, num_epochs + 1):
        torch.cuda.empty_cache()
        model.train()
        
        train_losses = []
        print(f'Epoch {epoch}/{num_epochs} - Training...')
        
        # Training phase with batching (exactly like DINOv2)
        for batch in tqdm(train_loader, desc='Training'):
            coordinates, targets = batch
            coordinates, targets = coordinates.to(device), targets.to(device)
            
            # Forward pass through SH + SIREN
            poverty_pred, representation = model(coordinates)
            loss = loss_fn(poverty_pred, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        mean_train_loss = np.mean(train_losses)
        torch.cuda.empty_cache()
        
        # Validation phase
        model.eval()
        val_losses = []
        indiv_losses = []
        print('Validating...')
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                coordinates, targets = batch
                coordinates, targets = coordinates.to(device), targets.to(device)
                
                # Forward pass
                poverty_pred, representation = model(coordinates)
                
                batch_loss = loss_fn(poverty_pred, targets)
                val_losses.append(batch_loss.item())
                indiv_losses.append(torch.mean(torch.abs(poverty_pred - targets), axis=0))
        
        # Compute mean validation loss
        mean_val_loss = np.mean(val_losses)   
        mean_indiv_loss = torch.stack(indiv_losses).mean(dim=0) if indiv_losses else torch.zeros(len(predict_target))
        
        # Learning rate scheduling
        scheduler.step(mean_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Save best model - following your pattern
        if mean_val_loss < best_error:
            print(f'New best model! Val loss: {mean_val_loss:.6f} < {best_error:.6f}')
            save_checkpoint(model, optimizer, epoch, mean_val_loss, filename=best_model)
            
            # Save additional metadata for SH + SIREN
            checkpoint = torch.load(best_model)
            checkpoint.update({
                'sh_L': sh_L,
                'representation_dim': representation_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'predict_target': predict_target,
                'target_dim': len(predict_target),
                'enhanced_targets': enhanced_targets,
                'country': country,
                'fold': fold
            })
            torch.save(checkpoint, best_model)
            best_error = mean_val_loss
        
        print(f'Epoch {epoch}: Train={mean_train_loss:.6f}, Val={mean_val_loss:.6f}, LR={current_lr:.6f}')
        print(f'  Individual losses (first 5): {mean_indiv_loss[:5].tolist()}')
        
        # Save last model - following your pattern
        save_checkpoint(model, optimizer, epoch, mean_val_loss, filename=last_model)
        
        # Save additional metadata for last model too
        checkpoint = torch.load(last_model)
        checkpoint.update({
            'sh_L': sh_L,
            'representation_dim': representation_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'predict_target': predict_target,
            'target_dim': len(predict_target),
            'enhanced_targets': enhanced_targets,
            'country': country,
            'fold': fold
        })
        torch.save(checkpoint, last_model)
        
        # Early stopping if learning rate gets too small
        if current_lr < 1e-7:
            print(f"Learning rate too small ({current_lr}), stopping early")
            break

    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_error:.6f}")
    print(f"Model trained on {len(train)} locations (same as DINOv2)")
    print(f"SH + SIREN features: {model.sh_encoder.output_dim} → {representation_dim}")

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
    parser.add_argument('--sh_L', type=int, default=15, help='Spherical harmonics degree (L parameter)')

    args = parser.parse_args()

    # Validate country code if provided
    if args.country:
        args.country = args.country.upper()
        if len(args.country) != 2:
            raise ValueError("Country code must be exactly 2 letters (e.g., ET, KE)")
        
    main(args.fold, args.target, args.imagery_path, args.imagery_source, args.representation_dim, 
         args.hidden_dim, args.num_layers, args.batch_size, args.num_epochs, args.country, 
         args.enhanced_targets, args.sh_L)