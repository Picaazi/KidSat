# Evulation Code from KidSat
import argparse
import numpy as np
import rasterio
import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
from tqdm import tqdm
import os

# Function to evaluate the model performance on given data
# fold: Fold number or country name
# model_name: model architecture like dinov2_vitb14
# target: prediction target (default is empty)
# use_checkpoint: whether to use a saved model
# model_not_named_target: helps form checkpoint filename
# imagery_path: path where satellite images are stored
# imagery_source: 'L' for Landsat, 'S' for Sentinel
# mode: evaluation type (temporal/spatial/one_country)
# model_output_dim: output feature size of model
# grouped_bands: which RGB bands to use for input image


def evaluate(fold, model_name, target="", use_checkpoint=False, model_not_named_target=True, imagery_path=None, 
            imagery_source=None, mode="temporal", model_output_dim=768, grouped_bands=None):
    
    model_par_dir = "modelling/dino/model/"

    # Build checkpoint filename (pth file) based on mode and target
    if use_checkpoint:
        named_target = target if model_not_named_target else ""
        if mode == "temporal":
            checkpoint = f"{model_par_dir}{model_name}_temporal_best_{imagery_source}{named_target}_.pth"
        elif mode == "spatial":
            checkpoint = f"{model_par_dir}{model_name}_{fold}_{grouped_bands}all_cluster_best_{imagery_source}{named_target}_.pth"
        elif mode == "one_country":
            checkpoint = f"{model_par_dir}{model_name}_{fold}_one_country_best_{imagery_source}{named_target}_.pth"
        else:
            raise Exception(mode)

    print(
        f"Evaluating {model_name} on fold {fold} with target {target} using checkpoint {checkpoint if use_checkpoint else 'None'}"
    )

    # Determine size of target
    if target == "":
        eval_target = "deprived_sev"
        target_size = 99
    else:
        eval_target = target
        target_size = 1 if model_not_named_target else 99

    # Image Modify based on the Satellite used (L and S)
    normalization = 30000.0 if imagery_source == "L" else 3000.0
    transform_dim = 336 if imagery_source == "L" else 994

    # DHS data folder
    data_folder = "survey_processing/processed_data/"

    # Termporal and Spatial have a different train test spilt
    if mode == "temporal":
        train_df = pd.read_csv(f"{data_folder}before_2020.csv")
        test_df = pd.read_csv(f"{data_folder}after_2020.csv")
    else:
        train_df = pd.read_csv(f"{data_folder}train_fold_{fold}.csv")
        test_df = pd.read_csv(f"{data_folder}test_fold_{fold}.csv")

    # Filter out imagery files that match the source type (L or S)
    available_imagery = [
        os.path.join(imagery_path, d, f)
        for d in os.listdir(imagery_path)
        if d[-2] == imagery_source
        for f in os.listdir(os.path.join(imagery_path, d))
    ]

    def is_available(centroid_id):
        return any(centroid_id in centroid for centroid in available_imagery)

    train_df = train_df[train_df["CENTROID_ID"].apply(is_available)]
    test_df = test_df[test_df["CENTROID_ID"].apply(is_available)]
    if test_df.empty:
        raise Exception("Empty test set")

    # Find the imagery file based on the CENTROID_ID
    def filter_contains(query):
        for item in available_imagery:
            if query in item:
                return item

    train_df["imagery_path"] = train_df["CENTROID_ID"].apply(filter_contains)
    train_df = train_df[train_df["deprived_sev"].notna()]
    test_df["imagery_path"] = test_df["CENTROID_ID"].apply(filter_contains)
    test_df = test_df[test_df["deprived_sev"].notna()]

    # Load image files and preprocess (stack bands, normalize, clip)
    def load_and_preprocess_image(path):
        with rasterio.open(path) as src:
            r = src.read(grouped_bands[0])
            g = src.read(grouped_bands[1])
            b = src.read(grouped_bands[2])
            img = np.dstack((r, g, b))
            img = img / normalization * 255.0
        img = np.nan_to_num(img, nan=0, posinf=255, neginf=0)
        return np.clip(img, 0, 255).astype(np.uint8)

    # Data are all prepared, now it is time for testing the model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained model from Facebook DINO hub
    base_model = (
        torch.hub.load("facebookresearch/dino:main", model_name)
        if "dino_" in model_name
        else torch.hub.load("facebookresearch/dinov2", model_name)
    )

    # Wrapper class to add regression head on top of DINO model
    class ViTForRegression(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.regression_head = nn.Linear(model_output_dim, target_size)

        def forward(self, pixel_values):
            outputs = self.base_model(pixel_values)
            return torch.sigmoid(self.regression_head(outputs))

        def forward_encoder(self, pixel_values):
            return self.base_model(pixel_values)

    model = ViTForRegression(base_model)

    # No idea how checkpoint is used, if known pls help me to add some comment
    if use_checkpoint:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict["model_state_dict"])

    # Dataset class for training and testing
    class CustomDataset(Dataset):
        def __init__(self, dataframe, transform):
            self.dataframe = dataframe
            self.transform = transform

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            item = self.dataframe.iloc[idx]
            image = load_and_preprocess_image(item["imagery_path"])
            image_tensor = self.transform(Image.fromarray(image))
            return image_tensor, item[eval_target]

    # Transform image to proper shape and type for model
    transform = transforms.Compose(
        [transforms.Resize((transform_dim, transform_dim)), transforms.ToTensor()]
    )

    # Dataloader wraps dataset to allow batch loading
    train_dataset = CustomDataset(train_df, transform)
    val_dataset = CustomDataset(test_df, transform)

    # I believe there is a certain format for the class CustomDataset in order to run
    # the function DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    model.to(device)
    model.eval()

    # Extract features from base model for training data
    X_train, y_train = [], []
    for images, targets in tqdm(train_loader):
        images, targets = images.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model.base_model(images)
        X_train.append(outputs.cpu()[0].numpy())
        y_train.append(targets.cpu()[0].numpy())

    # Extract features from base model for test data
    X_test, y_test = [], []
    for images, targets in tqdm(val_loader):
        images, targets = images.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model.base_model(images)
        X_test.append(outputs.cpu()[0].numpy())
        y_test.append(targets.cpu()[0].numpy())

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Save extracted features and targets to CSV
    results_folder = (
        f"modelling/dino/results/split_{mode}{imagery_source}_{fold}_{grouped_bands}/"
    )
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    pd.DataFrame(X_train).to_csv(results_folder + "X_train.csv", index=False)
    pd.DataFrame(y_train, columns=["target"]).to_csv(
        results_folder + "y_train.csv", index=False
    )
    pd.DataFrame(X_test).to_csv(results_folder + "X_test.csv", index=False)
    pd.DataFrame(y_test, columns=["target"]).to_csv(
        results_folder + "y_test.csv", index=False
    )

    # Ridge Regression with cross-validation to evaluate features
    alphas = np.logspace(-6, 6, 20)
    ridge_pipeline = Pipeline(
        [("ridge", RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_absolute_error"))]
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        ridge_pipeline, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error"
    )

    print("Cross-validation scores (negative MAE):", cv_scores)
    print("Mean cross-validation score (negative MAE):", cv_scores.mean())

    ridge_pipeline.fit(X_train, y_train)
    test_score = np.mean(np.abs(ridge_pipeline.predict(X_test) - y_test))
    print("Test Score (negative MAE):", test_score)

    return test_score



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run satellite image processing model training.')
    parser.add_argument('--fold', type=str, default='1', help='The fold number')
    parser.add_argument('--model_name', type=str, default='dinov2_vitb14', help='The model name')
    parser.add_argument('--target', type=str,default='', help='The target variable')
    parser.add_argument('--imagery_source', type=str, default='L', help='L for Landsat and S for Sentinel')
    parser.add_argument('--imagery_path', type=str, help='The parent directory of all imagery')
    parser.add_argument('--mode', type=str, default='temporal', help='Evaluating temporal model or spatial model')
    parser.add_argument('--model_output_dim', type=int, default=768, help='The output dimension of the model')
    parser.add_argument('--use_checkpoint', action='store_true', help='Whether to use checkpoint file. If not, use raw model.')
    parser.add_argument('--model_not_named_target', action='store_false', help='Whether the model name contains the target variable')
    parser.add_argument('--grouped_bands', nargs='+', type=int, help="List of grouped bands")
    
    args = parser.parse_args()
    maes = []
    if args.mode == 'temporal':
        print(evaluate("1", args.model_name,args.target, args.use_checkpoint,args.model_not_named_target, args.imagery_path, args.imagery_source, args.mode,  args.model_output_dim))
    elif args.mode == 'spatial':
        for i in range(5):
            fold = i + 1
            mae = evaluate(str(fold), args.model_name, args.target, args.use_checkpoint,args.model_not_named_target,args.imagery_path, args.imagery_source, args.mode, args.model_output_dim, grouped_bands=args.grouped_bands)
            maes.append(mae)
        print(np.mean(maes), np.std(maes)/np.sqrt(5))
    elif args.mode == 'one_country':
        COUNTRIES = ['Madagascar', 'Burundi', 'Uganda', 'Mozambique', 'Rwanda',
                    'Zambia', 'Tanzania', 'Malawi', 'Ethiopia', 'Kenya', 'Zimbabwe',
                    'Lesotho', 'South Africa', 'Angola', 'Eswatini', 'Comoros']
        
        n_samples = len(COUNTRIES)
        for country in COUNTRIES:
            try:
                mae = evaluate(country, args.model_name, args.target, args.use_checkpoint,args.model_not_named_target,args.imagery_path, args.imagery_source, args.mode, args.model_output_dim)
                maes.append(mae)
            except Exception as e:
                print(f"Error in {country}: {e}")
                n_samples -= 1
        print(np.mean(maes), np.std(maes)/np.sqrt(n_samples))
    else:
        raise Exception("Invalid mode")