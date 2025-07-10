import torch
import torch.nn as nn
import numpy as np
from scipy.special import sph_harm
from torch.utils.data import Dataset
import math
import warnings
warnings.filterwarnings('ignore')
import os
    
class ClippedReLU(nn.Module):
    def __init__(self, max_value=1.0):
        super(ClippedReLU, self).__init__()
        self.max_value = max_value

    def forward(self, x):
        return torch.clamp(x, min=0, max=self.max_value)
    
class ViTForRegression(nn.Module):
        def __init__(self, base_model, projection= None, activation="sigmoid", emb_size=768, predict_size=99):
            super().__init__()
            self.base_model = base_model
            if projection:
                self.projection = projection
            else:
                self.projection = nn.Identity()
                
            self.regression_head = nn.Linear(emb_size, predict_size)  # Output one continuous variable
            
            # Use sigmoid activation if specified, otherwise use ClippedReLU (sigmoid is inputed in the command line)
            if activation == "sigmoid":
                self.activation = nn.Sigmoid()
            elif activation == "clipped_relu":
                self.activation = ClippedReLU()
            else:
                self.activation = nn.Identity()
                
        def forward(self, pixel_values):
            outputs = self.base_model(self.projection(pixel_values))
            # We use the last hidden state
            return self.activation(self.regression_head(outputs))
        
class ViTForRegressionWithUncertainty(nn.Module):
    
        '''
                          [13-band Satellite Image]
                                │
               ┌────────────────┴────────────────┐
               │               ...               │
      [Band Group 1: 4,3,2]              [Band Group N: ...]
               │                                 |
         ViT Base Model 1                    ViT Base Model N
              │                                  │
             Feature 1                       Feature N
               └────────────────┬────────────────┘
                                ▼
                    [Cross-Attention (Multi-Head)]
                                ▼
                [Concat Attended Features: (B, N × emb_size)]
                                ▼
                    [Regression Head: Linear Layer]
                                ▼
                ┌─────────────────────────────┐
                │     Mean (Prediction)       │
                │     Variance (Uncertainty)  │
                └─────────────────────────────┘

    '''
        def __init__(self, base_models, grouped_bands=None, emb_size=768, predict_size=1):
            
            super().__init__()
            
            if grouped_bands is None:
                grouped_bands = [[0,1,2], [3,4,5], [6,7,8]]
                # grouped_bands = [[4, 3, 2], [8, 4, 2], [13, 1, 3], [12, 8, 2]]
                
            self.base_models = nn.ModuleList(base_models)
            self.grouped_bands = torch.tensor(grouped_bands) - 1
            self.cross_attention = nn.MultiheadAttention(embed_dim=emb_size, num_heads=8)
            
            # Update the regression head to output both mean and uncertainty
            # The output size is doubled to handle both prediction (mean) and log variance
            self.regression_head = nn.Linear(emb_size * len(grouped_bands), predict_size * 2)

        def forward(self, pixel_values):
            # Extract outputs from each base model with specific band groups
            outputs = [self.base_models[i](pixel_values[:, self.grouped_bands[i], :, :]) for i in range(len(self.base_models))]
            
            # Stack and permute outputs for multihead attention
            outputs = torch.stack(outputs, dim=0)  # Shape: [num_views, batch_size, emb_size]
            
            # Apply cross-attention
            attn_output, _ = self.cross_attention(outputs, outputs, outputs)  # Shape: [num_views, batch_size, emb_size]
            
            # Concatenate the attention output across all views
            concat_output = torch.cat([attn_output[i] for i in range(attn_output.size(0))], dim=-1)  # Shape: [batch_size, emb_size * num_views]
            
            # Pass through regression head to get mean and log variance
            regression_output = self.regression_head(concat_output)  # Shape: [batch_size, predict_target * 2]
            
            # Split the output into mean and log variance
            mean, log_var = torch.chunk(regression_output, 2, dim=-1)  # Each is of shape [batch_size, predict_target]
            
            # Calculate variance and uncertainty (variance must be positive, so apply exp)
            variance = torch.exp(log_var)  # Shape: [batch_size, predict_target]
            
            return mean, variance
        
'''
class SphericalHarmonicsEncoder:
    """
    Spherical Harmonics encoder following Russwurm & Korner (2023) implementation
    """
    def __init__(self, L=15):
        self.L = L
        self.output_dim = (L + 1) ** 2  # Real-valued SH
        print(f"SH Encoder: L={L}, output_dim={self.output_dim}")
    
    def associated_legendre_polynomial(self, l, m, x):
        """Associated Legendre polynomial implementation"""
        pmm = torch.ones_like(x)
        if m > 0:
            somx2 = torch.sqrt((1 - x) * (1 + x))
            fact = 1.0
            for i in range(1, m + 1):
                pmm = pmm * (-fact) * somx2
                fact += 2.0
        if l == m:
            return pmm
        pmmp1 = x * (2.0 * m + 1.0) * pmm
        if l == m + 1:
            return pmmp1
        pll = torch.zeros_like(x)
        for ll in range(m + 2, l + 1):
            pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
            pmm = pmmp1
            pmmp1 = pll
        return pll
    
    def SH_renormalization(self, l, m):
        """Spherical harmonics normalization factor"""
        return math.sqrt((2.0 * l + 1.0) * math.factorial(l - abs(m)) / \
            (4 * math.pi * math.factorial(l + abs(m))))
    
    def SH(self, m, l, phi, theta):
        """Compute spherical harmonics following Russwurm & Korner implementation"""
        if m == 0:
            return self.SH_renormalization(l, m) * \
                   self.associated_legendre_polynomial(l, m, torch.cos(theta))
        elif m > 0:
            return math.sqrt(2.0) * self.SH_renormalization(l, m) * \
                   torch.cos(m * phi) * self.associated_legendre_polynomial(l, m, torch.cos(theta))
        else:
            return math.sqrt(2.0) * self.SH_renormalization(l, -m) * \
                   torch.sin(-m * phi) * self.associated_legendre_polynomial(l, -m, torch.cos(theta))
    
    def encode_coordinates(self, lat, lon):
        """
        Encode lat/lon using spherical harmonics (Russwurm & Korner implementation)
        Returns: numpy array of shape (n_samples, output_dim)
        """
        # Convert to torch tensors
        lat = torch.tensor(lat, dtype=torch.float32)
        lon = torch.tensor(lon, dtype=torch.float32)
        
        # Convert to spherical coordinates
        theta = torch.deg2rad(90 - lat)  # Colatitude
        phi = torch.deg2rad(lon)         # Azimuth
        
        harmonics_features = []
        
        for l in range(self.L + 1):
            for m in range(-l, l + 1):
                sh_value = self.SH(m, l, phi, theta)
                harmonics_features.append(sh_value.numpy())
        
        # Stack features: (output_dim, n_samples) -> (n_samples, output_dim)
        features_array = np.stack(harmonics_features, axis=1)
        
        # Handle any numerical issues
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features_array

'''    
class SphericalHarmonicsEncoder:
    """
    Spherical Harmonics encoder for coordinates
    """
    def __init__(self, L=15):  # Reduced from 20 to prevent numerical issues
        self.L = L
        self.output_dim = self._calculate_output_dim()
        print(f"SH Encoder: L={L}, output_dim={self.output_dim}")
    
    def _calculate_output_dim(self):
        """Calculate output dimension based on L"""
        dim = 0
        for l in range(self.L + 1):
            for m in range(-l, l + 1):
                dim += 2  # Real and imaginary parts
        return dim
    
    def encode_coordinates(self, lat, lon):
        """
        Encode lat/lon using spherical harmonics
        Returns: numpy array of shape (n_samples, output_dim)
        """
        lat = np.atleast_1d(lat)
        lon = np.atleast_1d(lon)
        
        # Convert to spherical coordinates (physics convention)
        # theta = colatitude (0 to pi), phi = longitude (-pi to pi)
        theta = np.radians(90 - lat)  # Convert latitude to colatitude
        phi = np.radians(lon)
        
        harmonics_features = []
        
        for i in range(len(lat)):
            features_i = []
            for l in range(self.L + 1):
                for m in range(-l, l + 1):
                    try:
                        # Use correct spherical coordinate convention
                        Y_lm = sph_harm(m, l, phi[i], theta[i])
                        features_i.extend([Y_lm.real, Y_lm.imag])
                    except (RuntimeError, ValueError, OverflowError):
                        # Handle numerical issues gracefully
                        features_i.extend([0.0, 0.0])
            harmonics_features.append(features_i)
        
        features_array = np.array(harmonics_features)
        
        # Check for NaN/Inf and replace with zeros
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features_array


class PovertySirenSH(nn.Module):
    """
    Enhanced Siren network with Spherical Harmonics preprocessing
    Input: (lat, lon) coordinates → SH features → SIREN → poverty prediction
    """

    def __init__(self, sh_L=15, hidden_dim=256, num_layers=4,
                 representation_dim=128, omega_0=30.0):
        super().__init__()
        
        self.omega_0 = omega_0
        self.representation_dim = representation_dim
        
        # Spherical Harmonics encoder
        self.sh_encoder = SphericalHarmonicsEncoder(L=sh_L)
        sh_input_dim = self.sh_encoder.output_dim
        
        print(f"PovertySirenSH: SH input dim={sh_input_dim}, hidden={hidden_dim}, repr={representation_dim}")
        
        # SIREN encoder layers (SH features → representation)
        self.encoder_layers = nn.ModuleList()
        
        # First layer (SH features → hidden)
        first_layer = nn.Linear(sh_input_dim, hidden_dim)
        self.encoder_layers.append(first_layer)
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layer = nn.Linear(hidden_dim, hidden_dim)
            self.encoder_layers.append(layer)
        
        # Representation layer
        self.representation_layer = nn.Linear(hidden_dim, representation_dim)
        
        # Prediction head (will be replaced during training based on target size)
        self.prediction_head = nn.Sequential(
            nn.Linear(representation_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Will be adjusted based on target
            nn.Sigmoid()
        )
        
        # Initialize weights according to SIREN paper
        self.init_siren_weights()
    
    def init_siren_weights(self):
        """Initialize weights for SIREN network"""
        with torch.no_grad():
            # First layer: uniform distribution based on input dimension
            bound = 1 / self.encoder_layers[0].in_features
            self.encoder_layers[0].weight.uniform_(-bound, bound)
            
            # Hidden layers: scaled by sqrt(6/fan_in) / omega_0
            for layer in self.encoder_layers[1:]:
                bound = np.sqrt(6 / layer.in_features) / self.omega_0
                layer.weight.uniform_(-bound, bound)
            
            # Representation layer
            bound = np.sqrt(6 / self.representation_layer.in_features) / self.omega_0
            self.representation_layer.weight.uniform_(-bound, bound)
    
    def preprocess_coordinates(self, coords):
        """
        Convert coordinates to spherical harmonics features
        coords: torch tensor of shape (batch_size, 2) [lat, lon]
        """
        # Convert to numpy for SH encoding
        if coords.is_cuda:
            coords_np = coords.cpu().numpy()
        else:
            coords_np = coords.numpy()
        
        lat = coords_np[:, 0]
        lon = coords_np[:, 1]
        
        # Encode using spherical harmonics
        sh_features = self.sh_encoder.encode_coordinates(lat, lon)
        
        # Convert back to tensor on same device
        device = coords.device
        return torch.FloatTensor(sh_features).to(device)
    
    def encode_coordinates(self, coords):
        """
        Encode coordinates to learned representation
        coords: (batch_size, 2) [lat, lon]
        """
        # Preprocess coordinates to SH features
        sh_features = self.preprocess_coordinates(coords)
        
        x = sh_features
        
        # First layer with sine activation
        x = torch.sin(self.omega_0 * self.encoder_layers[0](x))
        
        # Hidden layers with sine activations
        for layer in self.encoder_layers[1:]:
            x = torch.sin(self.omega_0 * layer(x))
        
        # Get representation (no activation)
        representation = self.representation_layer(x)
        
        return representation
    
    def forward(self, coords):
        """
        Full forward pass: coordinates → SH → SIREN → poverty prediction
        """
        # Get learned representation
        representation = self.encode_coordinates(coords)
        
        # Predict poverty from representation
        poverty_pred = self.prediction_head(representation)
        
        return poverty_pred, representation

# Dataset class for SH + Siren training
class SirenSHDataset(Dataset):
    """Enhanced Dataset for SH + Siren coordinate-based training"""
    def __init__(self, dataframe, predict_target, coord_cols=['LATNUM', 'LONGNUM']):
        self.dataframe = dataframe
        self.predict_target = predict_target
        self.coord_cols = coord_cols
        
        # Check for required columns
        missing_cols = [col for col in coord_cols + predict_target if col not in dataframe.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Extract coordinates (no scaling needed for SH)
        self.coordinates = np.column_stack([
            dataframe[coord_cols[0]].values,  # LATNUM
            dataframe[coord_cols[1]].values   # LONGNUM
        ])
        
        # Handle multi-target case
        if isinstance(predict_target, list):
            self.targets = dataframe[predict_target].values.astype(np.float32)
            self.multi_target = True
        else:
            self.targets = dataframe[predict_target].values.astype(np.float32)
            self.multi_target = False
        
        # Remove rows with NaN coordinates or targets
        valid_mask = ~(np.isnan(self.coordinates).any(axis=1) | np.isnan(self.targets).any(axis=1) if self.multi_target else np.isnan(self.targets))
        
        self.coordinates = self.coordinates[valid_mask]
        self.targets = self.targets[valid_mask]
        
        print(f"SirenSHDataset: {len(self)} valid samples, target_dim={self.targets.shape[1] if self.multi_target else 1}")
        
    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        # Get coordinates as tensor (raw coordinates, not scaled)
        coords_tensor = torch.tensor(self.coordinates[idx], dtype=torch.float32)
        
        # Get target as tensor
        if self.multi_target:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
        else:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        return coords_tensor, target

# Function to load trained model
def load_sh_siren_model(model_path, device='cpu'):
    """Load trained SH + Siren model"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Reconstruct model
    model = PovertySirenSH(
        sh_L=checkpoint.get('sh_L', 15),
        hidden_dim=checkpoint['hidden_dim'],
        representation_dim=checkpoint['representation_dim'],
        omega_0=checkpoint.get('omega_0', 30.0)
    )
    
    # Adjust prediction head
    target_dim = checkpoint.get('target_dim', 1)

    model.prediction_head = nn.Sequential(
        nn.Linear(checkpoint['representation_dim'], 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, target_dim),
        nn.Sigmoid()
    )


    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    return model, checkpoint

# Function to extract features using trained model
def extract_sh_siren_features(model, coordinates, device='cpu', batch_size=64):
    """
    Extract location features using trained SH + Siren model
    
    Args:
        model: Trained PovertySirenSH model
        coordinates: numpy array of shape (n_samples, 2) [lat, lon]
        device: torch device
        batch_size: batch size for processing
    
    Returns:
        numpy array of location features
    """
    model.eval()
    model.to(device)
    
    features_list = []
    
    # Process in batches to handle memory constraints
    for i in range(0, len(coordinates), batch_size):
        batch_coords = coordinates[i:i+batch_size]
        coords_tensor = torch.FloatTensor(batch_coords).to(device)
        
        with torch.no_grad():
            batch_features = model.encode_coordinates(coords_tensor)
            features_list.append(batch_features.cpu().numpy())
    
    return np.concatenate(features_list, axis=0)