import torch
import torch.nn as nn

    
class ClippedReLU(nn.Module):
    def __init__(self, max_value=1.0):
        super(ClippedReLU, self).__init__()
        self.max_value = max_value

    def forward(self, x):
        return torch.clamp(x, min=0, max=self.max_value)
    
class ViTForRegression(nn.Module):
        def __init__(self, base_model, projection= None, activation="sigmoid", emb_size=768, predict_target=99):
            super().__init__()
            self.base_model = base_model
            if projection:
                self.projection = projection
            else:
                self.projection = nn.Identity()
            # Assuming the original model outputs 768 features from the transformer
            self.regression_head = nn.Linear(emb_size, predict_target)  # Output one continuous variable
            
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
        def __init__(self, base_models, grouped_bands=[[4, 3, 2], [8, 4, 2], [13, 1, 3], [12, 8, 2]], emb_size=768, predict_target=1):
            super().__init__()
            self.base_models = nn.ModuleList(base_models)
            self.grouped_bands = torch.tensor(grouped_bands) - 1
            self.cross_attention = nn.MultiheadAttention(embed_dim=emb_size, num_heads=8)
            
            # Update the regression head to output both mean and uncertainty
            # The output size is doubled to handle both prediction (mean) and log variance
            self.regression_head = nn.Linear(emb_size * len(grouped_bands), predict_target * 2)

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