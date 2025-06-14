import argparse
import numpy as np
import rasterio
import numpy as np
import torch
from torch import nn
import os
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset, DataLoader
import numpy as np
from models import ViTForRegressionWithUncertainty
from preparation import image_config, set_seed, CustomDataset, get_datasets
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
from tqdm import tqdm

def evaluate(fold, model_name, target="", use_checkpoint=False, imagery_path=None, imagery_source=None, mode="temporal"):
    
    model_par_dir = r"modelling/dino/model/"

    import os

    best_model = (f"{model_name}ms_uncer_{fold}_all_cluster_best_{imagery_source}{target}_.pth")
    checkpoint = os.path.join(model_par_dir, best_model)
    
    normalization, imagery_size = image_config(imagery_source)

    print(f"Evaluating fold {fold} with target {target} using checkpoint {checkpoint}")

    # if imagery_source == "L":
    #     normalization = 30000.0
    #     transform_dim = 336
    # elif imagery_source == "S":
    #     normalization = 3000.0
    #     transform_dim = 994

    data_folder = r"survey_processing/processed_data/"
    
    
    if "spatial" in mode:
        train_df = pd.read_csv(f"{data_folder}train_fold_{fold}.csv")
        test_df = pd.read_csv(f"{data_folder}test_fold_{fold}.csv")
    elif "temporal" in mode:
        train_df = pd.read_csv(f"{data_folder}before_2020.csv")
        test_df = pd.read_csv(f"{data_folder}after_2020.csv")
    elif mode == "one_country":
        train_df = pd.read_csv(f"{data_folder}train_fold_{fold}.csv")
        test_df = pd.read_csv(f"{data_folder}test_fold_{fold}.csv")
        
    train_df, test_df, predict_target = get_datasets(train_df, test_df, imagery_path, imagery_source, target)

    # available_imagery = []

    # for d in os.listdir(imagery_path):
    #     if d[-2] == imagery_source:
    #         for f in os.listdir(os.path.join(imagery_path, d)):
    #             available_imagery.append(os.path.join(imagery_path, d, f))

    # def is_available(centroid_id):
    #     for centroid in available_imagery:
    #         if centroid_id in centroid:
    #             return True
    #     return False

    # train_df = train_df[train_df["CENTROID_ID"].apply(is_available)]
    # test_df = test_df[test_df["CENTROID_ID"].apply(is_available)]
    
    # if test_df.empty:
    #     raise Exception("Empty test set")

    # def filter_contains(query):
    #     """
    #     Returns a list of items that contain the given query substring.

    #     Parameters:
    #         items (list of str): The list of strings to search within.
    #         query (str): The substring to search for in each item of the list.

    #     Returns:
    #         list of str: A list containing all items that have the query substring.
    #     """
    #     # Use a list comprehension to filter items
    #     for item in available_imagery:
    #         if query in item:
    #             return item

    # train_df["imagery_path"] = train_df["CENTROID_ID"].apply(filter_contains)
    # train_df = train_df[train_df["deprived_sev"].notna()]
    # test_df["imagery_path"] = test_df["CENTROID_ID"].apply(filter_contains)
    # test_df = test_df[test_df["deprived_sev"].notna()]

    # def load_and_preprocess_image(path):
    #     with rasterio.open(path) as src:
    #         bands = src.read()
    #         img = bands[:13]
    #         img = img / normalization  # Normalize to [0, 1] (if required)

    #     img = np.nan_to_num(img, nan=0, posinf=1, neginf=0)
    #     img = np.clip(img, 0, 1)  # Clip values to be within the 0-1 range
    #     img = np.transpose(img, (1, 2, 0))
    #     # Scale back to [0, 255] for visualization purposes
    #     img = (img * 255).astype(np.uint8)

    #     return img

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_models = [torch.hub.load("facebookresearch/dinov2", model_name).to(device)for _ in range(3)]

    # class ViTForRegressionWithUncertainty(nn.Module):
    #     def __init__(self, base_models, grouped_bands=[[4, 3, 2], [8, 4, 2], [13, 1, 3], [12, 8, 2]], emb_size=768, predict_target=1):
    #         super().__init__()
    #         self.base_models = nn.ModuleList(base_models)
    #         self.grouped_bands = torch.tensor(grouped_bands) - 1
    #         self.cross_attention = nn.MultiheadAttention(embed_dim=emb_size, num_heads=8)

    #         # Update the regression head to output both mean and uncertainty
    #         # The output size is doubled to handle both prediction (mean) and log variance
    #         self.regression_head = nn.Linear(emb_size * len(grouped_bands), predict_target * 2)

    #     def forward(self, pixel_values):
    #         # Extract outputs from each base model with specific band groups
    #         outputs = [
    #             self.base_models[i](pixel_values[:, self.grouped_bands[i], :, :])
    #             for i in range(len(self.base_models))
    #         ]

    #         # Stack and permute outputs for multihead attention
    #         outputs = torch.stack(outputs, dim=0)  # Shape: [num_views, batch_size, emb_size]

    #         # Apply cross-attention
    #         attn_output, _ = self.cross_attention(outputs, outputs, outputs)  # Shape: [num_views, batch_size, emb_size]

    #         # Concatenate the attention output across all views
    #         concat_output = torch.cat([attn_output[i] for i in range(attn_output.size(0))], dim=-1)  # Shape: [batch_size, emb_size * num_views]

    #         # Pass through regression head to get mean and log variance
    #         regression_output = self.regression_head(concat_output)  # Shape: [batch_size, predict_target * 2]

    #         # Split the output into mean and log variance
    #         mean, log_var = torch.chunk(regression_output, 2, dim=-1)  # Each is of shape [batch_size, predict_target]

    #         # Calculate variance and uncertainty (variance must be positive, so apply exp)
    #         variance = torch.exp(log_var)  # Shape: [batch_size, predict_target]

    #         return mean, variance

    model = ViTForRegressionWithUncertainty(base_models).to(device)
    if use_checkpoint:
        print(f"Loading checkpoint from {checkpoint}")
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict["model_state_dict"])
    eval_target = target

    # class CustomDataset(Dataset):
    #     def __init__(self, dataframe, transform):
    #         self.dataframe = dataframe
    #         self.transform = transform

    #     def __len__(self):
    #         return len(self.dataframe)

    #     def __getitem__(self, idx):
    #         item = self.dataframe.iloc[idx]
    #         image = load_and_preprocess_image(item["imagery_path"])
    #         # Apply feature extractor if necessary, might need adjustments
    #         image_tensor = self.transform(image)

    #         # Assuming your target is a single scalar
    #         target = torch.tensor(item[eval_target], dtype=torch.float32)
    #         return (
    #             image_tensor,
    #             target,
    #         )  # Adjust based on actual output of feature_extractor

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Resize(
                (imagery_size, imagery_size)
            ),  # Resize the image to the input size expected by the model
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet's mean and std
        ]
    )

    train_dataset = CustomDataset(train_df, transform, normalization, predict_target, all=True)
    val_dataset = CustomDataset(test_df, transform, normalization, predict_target, all=True)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    model.to(device)
    model.eval()

    results_folder = (
        f"modelling/dino/results/split_{mode}_msuncer_{imagery_source}_{fold}/"
    )

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    X_train = []
    y_train = []
    for batch in tqdm(train_loader):
        images, targets = batch
        images, targets = images.to(device), targets.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(images)
        outputs = [o.cpu()[0].numpy()[0] for o in outputs]
        print(outputs, targets, "train")
        X_train.append(outputs)
        y_train.append(targets.cpu()[0].numpy())

    torch.cuda.empty_cache()
    # Validation phase
    X_test = []
    y_test = []
    for batch in tqdm(val_loader):
        images, targets = batch
        images, targets = images.to(device), targets.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(images)
        outputs = [o.cpu()[0].numpy()[0] for o in outputs]
        print(outputs, targets, "test")
        X_test.append(outputs)
        y_test.append(targets.cpu()[0].numpy())

    # Convert lists to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Convert to pandas DataFrames
    df_X_train = pd.DataFrame(X_train)
    df_y_train = pd.DataFrame(y_train, columns=["target"])
    df_X_test = pd.DataFrame(X_test)
    df_y_test = pd.DataFrame(y_test, columns=["target"])

    # Save to CSV files
    df_X_train.to_csv(results_folder + "X_train.csv", index=False)
    df_y_train.to_csv(results_folder + "y_train.csv", index=False)
    df_X_test.to_csv(results_folder + "X_test.csv", index=False)
    df_y_test.to_csv(results_folder + "y_test.csv", index=False)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run satellite image processing model training.")
    parser.add_argument("--fold", type=str, default="1", help="The fold number")
    parser.add_argument("--model_name", type=str, default="dinov2_vitb14", help="The model name")
    parser.add_argument("--target", type=str, default="", help="The target variable")
    parser.add_argument("--imagery_source",type=str,default="L",help="L for Landsat and S for Sentinel")
    parser.add_argument("--imagery_path", type=str, help="The parent directory of all imagery")
    parser.add_argument("--mode",type=str,default="temporal",help="Evaluating temporal model or spatial model")
    parser.add_argument("--use_checkpoint",action="store_true",help="Whether to use checkpoint file. If not, use raw model.")

    args = parser.parse_args()
    maes = []
    if args.mode == "temporal":
        print(evaluate("1", args.model_name, args.target, args.use_checkpoint, args.imagery_path, args.imagery_source, args.mode))
    elif "spatial" in args.mode:
        for i in range(5):
            fold = i + 1
            mae = evaluate(str(fold), args.model_name, args.target, args.use_checkpoint, args.imagery_path, args.imagery_source, args.mode)
            maes.append(mae)
        print("MAE for each fold:", maes)
        print(np.mean(maes), np.std(maes)/np.sqrt(5))
    else:
        raise Exception("Invalid mode")
