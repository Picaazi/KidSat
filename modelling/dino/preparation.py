import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import rasterio
import random
import os
import re

def get_datasets(train_df, test_df, imagery_path, imagery_source, target =''):
    available_imagery = []
    for d in os.listdir(imagery_path):
        if d[-2] == imagery_source:
            for f in os.listdir(os.path.join(imagery_path, d)):
                available_imagery.append(os.path.join(imagery_path, d, f))

    def is_available(centroid_id):
        for centroid in available_imagery:
            if centroid_id in centroid:
                return True
        return False
    
    train_df = train_df[train_df['CENTROID_ID'].apply(is_available)]
    test_df = test_df[test_df['CENTROID_ID'].apply(is_available)]
    
    def filter_contains(query):
        # Returns a list of items that contain the given query substring.

        # Use a list comprehension to filter items
        for item in available_imagery:
            if query in item:
                return item
    
    train_df['imagery_path'] = train_df['CENTROID_ID'].apply(filter_contains)
    test_df['imagery_path'] = test_df['CENTROID_ID'].apply(filter_contains)
    if target == '':
        predict_target = ['h10', 'h3', 'h31', 'h5', 'h7', 'h9', 
                        'hc70', 'hv109', 'hv121', 'hv106', 'hv201', 
                        'hv204', 'hv205', 'hv216', 'hv225', 'hv271', 'v312']
    else:
        predict_target = [target]

    filtered_predict_target = []
    for col in predict_target:
        filtered_predict_target.extend(
            [c for c in train_df.columns if c == col or re.match(f"^{col}_[^a-zA-Z]", c)]
        )
    # Drop rows with NaN values in the filtered subset of columns
    train_df = train_df.dropna(subset=filtered_predict_target)
    predict_target = sorted(filtered_predict_target)
    
    return train_df, test_df, predict_target
    

def image_config(imagery_source, img_size=None):
    if imagery_source == 'L':
        normalization = 30000.
        imagery_size = 336
    elif imagery_source == 'S':
        normalization = 3000.
        imagery_size = 994
    else:
        raise Exception("Unsupported imagery source")
    
    if not img_size is None:
        imagery_size = img_size
        
    return normalization, imagery_size

# Load and preprocess the image using selected bands (RGB)
def load_and_preprocess_image(path, normalization, grouped_bands):
    with rasterio.open(path) as src:
        b1 = src.read(grouped_bands[0])
        b2 = src.read(grouped_bands[1])
        b3 = src.read(grouped_bands[2])

        # Stack and normalize the bandss
        img = np.dstack((b1, b2, b3))
        img = img / normalization  # Normalize to [0, 1] (if required)

    img = np.nan_to_num(img, nan=0, posinf=1, neginf=0)
    img = np.clip(img, 0, 1)  # Clip values to be within the 0-1 range

    # Scale back to [0, 255] for visualization purposes
    img = (img * 255).astype(np.uint8)

    return img

def load_and_preprocess_image_all(path, normalization):
    
    new_order = [4,3,2,5,4,2]
    
    # input which band groups to use, then order them in the right order
    with rasterio.open(path) as src:
        bands = src.read()
        img = bands[:13]
        img = img / normalization  # Normalize to [0, 1] (if required)
    
    img = np.nan_to_num(img, nan=0, posinf=1, neginf=0)
    img = np.clip(img, 0, 1)  # Clip values to be within the 0-1 range
    img = np.transpose(img, (1, 2, 0))
    # Scale back to [0, 255] for visualization purposes
    img = (img * 255).astype(np.uint8)

    return img

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform, normalization, predict_target, grouped_bands=None, all = False):
        self.dataframe = dataframe
        self.transform = transform
        
        self.normalization = normalization
        self.predict_target = predict_target
        self.grouped_bands = grouped_bands
        self.all = all
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        path = item['imagery_path']
        gb = [4,3,2]
        if self.grouped_bands is None:
            if "L7" in path or "L5" in path:
                gb = [3, 2, 1] 
            elif "L8" in path or "S2" in path:
                gb = [4, 3, 2]
            else:
                print("No satilite found, idk what band to use")
        else:
            gb = self.grouped_bands
            
        if self.all:
            image = load_and_preprocess_image_all(path, self.normalization)
        else:
            image = load_and_preprocess_image(path, self.normalization, gb)
        # Apply feature extractor if necessary, might need adjustments
        image_tensor = self.transform(Image.fromarray(image))
        
        # Assuming your target is a single scalar                                                                                                                                                                                                                                                                                                                                                                                                      
        target = torch.tensor(item[self.predict_target], dtype=torch.float32)
        return image_tensor, target  # Adjust based on actual output of feature_extractor

# Function to save model checkpoints
def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, filename)
    print(f"Checkpoint saved to {filename}, with loss: {loss:.4f} at epoch {epoch}")