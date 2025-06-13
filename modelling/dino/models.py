import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
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
        def __init__(self, base_models, grouped_bands=[[4, 3, 2], [8, 4, 2], [13, 1, 3], [12, 8, 2]], emb_size=768, predict_size=1):
            super().__init__()
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
    
class PovertySiren(nn.Module):
    """
    Siren network trained specifically for poverty prediction
    Input: (lat, lon) coordinates
    Output: Learned location representation + poverty prediction
    """

    def __init__(self, input_dim=2, hidden_dim=256, num_layers=4,
                 representation_dim=128, omega_0=30.0):
        super().__init__()
        
        self.omega_0 = omega_0
        self.representation_dim = representation_dim
        
        # Encoder layers (coordinate → representation)
        self.encoder_layers = nn.ModuleList()
        
        # First layer
        first_layer = nn.Linear(input_dim, hidden_dim)
        self.encoder_layers.append(first_layer)
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layer = nn.Linear(hidden_dim, hidden_dim)
            self.encoder_layers.append(layer)
        
        # Representation layer (this is what we'll extract as features)
        self.representation_layer = nn.Linear(hidden_dim, representation_dim)
        
        # Prediction head (representation → poverty score)
        self.prediction_head = nn.Sequential(
            nn.Linear(representation_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights according to SIREN paper
        self.init_siren_weights()
    
    def init_siren_weights(self):
        """Initialize weights for SIREN network"""
        with torch.no_grad():
            # First layer: uniform distribution
            bound = 1 / self.encoder_layers[0].in_features
            self.encoder_layers[0].weight.uniform_(-bound, bound)
            
            # Hidden layers: scaled by sqrt(6/fan_in) / omega_0
            for layer in self.encoder_layers[1:]:
                bound = np.sqrt(6 / layer.in_features) / self.omega_0
                layer.weight.uniform_(-bound, bound)
            
            # Representation layer
            bound = np.sqrt(6 / self.representation_layer.in_features) / self.omega_0
            self.representation_layer.weight.uniform_(-bound, bound)
    
    def encode_coordinates(self, coords):
        """
        Encode coordinates to learned representation
        This is the function we'll use for feature extraction
        """
        x = coords
        
        # First layer with sine activation
        x = torch.sin(self.omega_0 * self.encoder_layers[0](x))
        
        # Hidden layers with sine activations
        for layer in self.encoder_layers[1:]:
            x = torch.sin(self.omega_0 * layer(x))
        
        # Get representation (without activation)
        representation = self.representation_layer(x)
        
        return representation
    
    def forward(self, coords):
        """
        Full forward pass: coordinates → representation → poverty prediction
        """
        # Get learned representation
        representation = self.encode_coordinates(coords)
        
        # Predict poverty from representation
        poverty_pred = self.prediction_head(representation)
        
        return poverty_pred, representation
    

# Add helper functions for loading Siren models
def load_siren_model(model_path, device='cpu'):
    """
    Helper function to load a trained Siren model from checkpoint
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Siren model not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model parameters
    representation_dim = checkpoint['representation_dim']
    hidden_dim = checkpoint['hidden_dim']
    num_layers = checkpoint['num_layers']
    predict_target = checkpoint['predict_target']
    
    # Reconstruct model
    model = PovertySiren(
        input_dim=2,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        representation_dim=representation_dim
    )
    
    # Modify prediction head to match training
    model.prediction_head = nn.Sequential(
        nn.Linear(representation_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, len(predict_target)),
        nn.Sigmoid()
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['coord_scaler']

def extract_siren_features(model, coord_scaler, coordinates, device='cpu'):
    """
    Helper function to extract Siren features from coordinates
    """
    # Scale coordinates
    coords_scaled = coord_scaler.transform(coordinates)
    coords_tensor = torch.FloatTensor(coords_scaled).to(device)
    
    # Extract features
    model.eval()
    with torch.no_grad():
        representations = model.encode_coordinates(coords_tensor)
    
    return representations.cpu().numpy()