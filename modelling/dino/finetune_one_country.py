# Importing necessary libraries
import argparse  # For command line argument parsing
import pandas as pd  # For data manipulation
from tqdm import tqdm  # For progress bars during loops
import os
import random
import rasterio  # For reading satellite imagery
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

warnings.filterwarnings("ignore")  # Ignore warnings for clean output


# Main function that handles training
def main(
    country,
    model_name,
    target,
    imagery_path,
    imagery_source,
    emb_size,
    batch_size,
    num_epochs,
    imagery_size=None,
):

    # Set normalization and default image size based on the satellite imagery source
    if imagery_source == "L":
        normalization = 30000.0
        imagery_size = 336
    elif imagery_source == "S":
        normalization = 3000.0
        imagery_size = 994
    else:
        raise Exception("Unsupported imagery source")

    if not imagery_size is None:
        imagery_size = imagery_size

    # Define where the data is stored
    data_folder = r"survey_processing/processed_data"

    # Read training and testing data from CSV files
    train_df = pd.read_csv(f"{data_folder}/train_fold_{country}.csv")
    test_df = pd.read_csv(f"{data_folder}/test_fold_{country}.csv")

    # List all available satellite images
    available_imagery = []
    for d in os.listdir(imagery_path):
        if d[-2] == imagery_source:
            for f in os.listdir(os.path.join(imagery_path, d)):
                available_imagery.append(os.path.join(imagery_path, d, f))

    # Filter out rows that don't have corresponding imagery
    def is_available(centroid_id):
        for centroid in available_imagery:
            if centroid_id in centroid:
                return True
        return False

    train_df = train_df[train_df["CENTROID_ID"].apply(is_available)]
    test_df = test_df[test_df["CENTROID_ID"].apply(is_available)]
    if test_df.empty:
        raise Exception(f"No test data available for {country}")

    # Find the exact image path for each centroid ID
    def filter_contains(query):
        """
        Returns a list of items that contain the given query substring.

        Parameters:
            items (list of str): The list of strings to search within.
            query (str): The substring to search for in each item of the list.

        Returns:
            list of str: A list containing all items that have the query substring.
        """
        # Use a list comprehension to filter items
        for item in available_imagery:
            if query in item:
                return item

    train_df["imagery_path"] = train_df["CENTROID_ID"].apply(filter_contains)
    test_df["imagery_path"] = test_df["CENTROID_ID"].apply(filter_contains)

    # If no specific target is provided, predict multiple default variables
    if target == "":
        predict_target = [
            "h10",
            "h3",
            "h31",
            "h5",
            "h7",
            "h9",
            "hc70",
            "hv109",
            "hv121",
            "hv106",
            "hv201",
            "hv204",
            "hv205",
            "hv216",
            "hv225",
            "hv271",
            "v312",
        ]
    else:
        predict_target = [target]

    # Get columns that match the prediction targets, even if postfixed (e.g., h10_1)
    filtered_predict_target = []
    for col in predict_target:
        filtered_predict_target.extend(
            [
                c
                for c in train_df.columns
                if c == col or re.match(f"^{col}_[^a-zA-Z]", c)
            ]
        )
    # Drop rows with NaN values in the filtered subset of columns
    train_df = train_df.dropna(subset=filtered_predict_target)
    predict_target = sorted(filtered_predict_target)

    # Load and preprocess the image using selected bands (RGB)
    def load_and_preprocess_image(path, grouped_bands=[4, 3, 2]):
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

    # Set random seed for reproducibility
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set your desired seed
    seed = 42
    set_seed(seed)

    # Split training data into train and validation sets
    train, validation = train_test_split(train_df, test_size=0.2, random_state=42)

    # Custom PyTorch dataset for loading image and target
    class CustomDataset(Dataset):
        def __init__(self, dataframe, transform):
            self.dataframe = dataframe
            self.transform = transform

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            item = self.dataframe.iloc[idx]
            image = load_and_preprocess_image(item["imagery_path"])
            # Apply feature extractor if necessary, might need adjustments
            image_tensor = self.transform(Image.fromarray(image))

            # Assuming your target is a single scalar
            target = torch.tensor(item[predict_target], dtype=torch.float32)
            return (
                image_tensor,
                target,
            )  # Adjust based on actual output of feature_extractor

    # Transform to resize and convert images to tensors
    transform = transforms.Compose(
        [
            transforms.Resize(
                (imagery_size, imagery_size)
            ),  # Resize the image to the input size expected by the model
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet's mean and std
        ]
    )

    # Create dataset and data loader
    train_dataset = CustomDataset(train, transform)
    val_dataset = CustomDataset(validation, transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size + 4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size + 4
    )

    # Load pretrained DINOv2 model from Facebook's repo
    base_model = torch.hub.load("facebookresearch/dinov2", model_name)

    # Function to save model checkpoints
    def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth"):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            filename,
        )

    torch.cuda.empty_cache()

    # Create a regression model using DINOv2 as feature extractor
    class ViTForRegression(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            # Assuming the original model outputs 768 features from the transformer
            self.regression_head = nn.Linear(
                emb_size, len(predict_target)
            )  # Output one continuous variable

        def forward(self, pixel_values):
            outputs = self.base_model(pixel_values)
            # We use the last hidden state
            return torch.sigmoid(self.regression_head(outputs))

    # Resume from checkpoint if exists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForRegression(base_model).to(device)
    best_model = f"modelling/dino/model/{model_name}_{country}_one_country_best_{imagery_source}{target}.pth"
    last_model = f"modelling/dino/model/{model_name}_{country}_one_country_last_{imagery_source}{target}.pth"
    if os.path.exists(last_model):
        last_state_dict = torch.load(last_model)
        best_error = torch.load(best_model)["loss"]
        epoch_ran = last_state_dict["epoch"]
        model.load_state_dict(last_state_dict["model_state_dict"])
        print("Found existing model")
    else:
        epochs_ran = 0
        best_error = np.inf

    # Move model to appropriate device
    model.to(device)

    # Set different learning rates for base model and head
    base_model_params = {
        "params": model.base_model.parameters(),
        "lr": 1e-6,
        "weight_decay": 1e-6,
    }
    head_params = {
        "params": model.regression_head.parameters(),
        "lr": 1e-6,
        "weight_decay": 1e-6,
    }

    # Setup the optimizer
    optimizer = torch.optim.Adam([base_model_params, head_params])
    loss_fn = L1Loss()

    # Training Loop
    for epoch in range(epochs_ran + 1, num_epochs):
        torch.cuda.empty_cache()
        model.train()
        print("Training...")
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
        print("Validating...")
        for batch in val_loader:
            images, targets = batch
            images, targets = images.to(device), targets.to(device)

            # Forward pass
            with torch.no_grad():
                outputs = model(images)
            batch_loss = loss_fn(outputs, targets)
            val_loss.append(batch_loss.item())
            indiv_loss.append(torch.mean(torch.abs(outputs - targets), axis=0))

        # Compute mean validation loss
        mean_val_loss = np.mean(val_loss)
        mean_indiv_loss = torch.stack(indiv_loss).mean(dim=0)

        if mean_val_loss < best_error:
            save_checkpoint(model, optimizer, epoch, mean_val_loss, filename=best_model)
            best_error = mean_val_loss
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {mean_val_loss}, Individual Loss: {mean_indiv_loss}"
        )
        save_checkpoint(model, optimizer, epoch, mean_val_loss, filename=last_model)


# Entry point when running from command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run satellite image processing model training."
    )
    parser.add_argument("--fold", type=str, help="CV fold")
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument("--target", type=str, default="", help="Target variable")
    parser.add_argument(
        "--imagery_path", type=str, help="The parent directory of all imagery"
    )
    parser.add_argument(
        "--imagery_source",
        type=str,
        default="L",
        help="L for Landsat and S for Sentinel",
    )
    parser.add_argument(
        "--emb_size", type=int, default=768, help="Size of the model output"
    )
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of epochs for training"
    )
    parser.add_argument("--imagery_size", type=int, help="Size of the imagery")
    args = parser.parse_args()
    main(
        args.fold,
        args.model_name,
        args.target,
        args.imagery_path,
        args.imagery_source,
        args.emb_size,
        args.batch_size,
        args.num_epochs,
        args.imagery_size,
    )
