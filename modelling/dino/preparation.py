import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import rasterio
import random

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


def load_and_preprocess_image(path, normalization, grouped_bands=[4, 3, 2]):
    with rasterio.open(path) as src:
        b1 = src.read(grouped_bands[0])
        b2 = src.read(grouped_bands[1])
        b3 = src.read(grouped_bands[2])

        # Stack and normalize the bands
        img = np.dstack((b1, b2, b3))
        img = img / normalization  # Normalize to [0, 1] (if required)

    img = np.nan_to_num(img, nan=0, posinf=1, neginf=0)
    img = np.clip(img, 0, 1)  # Clip values to be within the 0-1 range

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
    def __init__(self, dataframe, transform, normalization, predict_target):
        self.dataframe = dataframe
        self.transform = transform
        self.normalization = normalization
        self.predict_target = predict_target

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        image = load_and_preprocess_image(item['imagery_path'], self.normalization)
        # Apply feature extractor if necessary, might need adjustments
        image_tensor = self.transform(Image.fromarray(image))
        
        # Assuming your target is a single scalar
        target = torch.tensor(item[self.predict_target], dtype=torch.float32)
        return image_tensor, target  # Adjust based on actual output of feature_extractor
