# Swin Transformer Evaluation Script for KidSat (torchvision compatible)
import argparse
import numpy as np
import rasterio
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset, DataLoader
import warnings
import sys
import os

warnings.filterwarnings("ignore")
import pandas as pd
from tqdm import tqdm

# Import your custom modules
sys.path.append('/user/work/cy23765/kidsat_test/KidSat/modelling/dino')
from preparation import get_datasets
from models import ViTForRegression

def evaluate_swin(
    fold,
    model_name,
    target="",
    use_checkpoint=False,
    model_not_named_target=True,
    imagery_path=None,
    imagery_source=None,
    mode="temporal",
    model_output_dim=1024,
    grouped_bands=None,
    emb_size=1000,
):
    """
    Evaluate Swin transformer model performance on satellite imagery
    Compatible with torchvision models and your training setup
    
    Args:
        fold: Fold number or country name
        model_name: Swin model name ('swin_b', 'swin_s', 'swin_t')
        target: prediction target (default is empty for severe deprivation)
        use_checkpoint: whether to use a saved model
        model_not_named_target: helps form checkpoint filename
        imagery_path: path where satellite images are stored
        imagery_source: 'L' for Landsat, 'S' for Sentinel
        mode: evaluation type (temporal/spatial/one_country)
        model_output_dim: output feature size of model
        grouped_bands: which bands to use for input image
        emb_size: embedding size from your training script
    """
    
    model_par_dir = "modelling/dino/model/"  # Using same path as your training script
    
    # Build checkpoint filename based on mode and target
    if use_checkpoint:
        if mode == "temporal":
            checkpoint = f"{model_par_dir}{model_name}_temporal_best_{imagery_source}_.pth"
        elif mode == "spatial":
            checkpoint = f"{model_par_dir}{model_name}_{fold}_all_cluster_best_{imagery_source}_.pth"
        elif mode == "one_country":
            checkpoint = f"{model_par_dir}{model_name}_{fold}_one_country_best_{imagery_source}_.pth"
        else:
            raise Exception(f"Invalid mode: {mode}")

    print(
        f"Evaluating {model_name} on fold {fold} with target {target} using checkpoint {checkpoint if use_checkpoint else 'None'}"
    )

    # Image normalization based on satellite type (matching your training script)
    if imagery_source == 'L':
        normalization = 30000.0
        imagery_size = 336
    elif imagery_source == 'S':
        normalization = 3000.0
        imagery_size = 1024
    else:
        raise Exception("Unsupported imagery source")

    # DHS data folder
    data_folder = "survey_processing/processed_data/"

    # Load train/test split based on mode
    if mode == "temporal":
        train_df = pd.read_csv(f"{data_folder}before_2020.csv")
        test_df = pd.read_csv(f"{data_folder}after_2020.csv")
    else:
        train_df = pd.read_csv(f"{data_folder}train_fold_{fold}.csv")
        test_df = pd.read_csv(f"{data_folder}test_fold_{fold}.csv")

    # Use your existing get_datasets function
    train_df, test_df, predict_target = get_datasets(train_df, test_df, imagery_path, imagery_source, target)
    
    if test_df.empty:
        raise Exception("Empty test set")

    # ALWAYS use 99 targets to match trained model, regardless of what target is specified
    if target == "":
        eval_target = predict_target
        target_size = len(predict_target)
    else:
        # Force get all targets even when specific target is requested
        temp_train_df, temp_test_df, all_predict_target = get_datasets(train_df, test_df, imagery_path, imagery_source, "")
        eval_target = all_predict_target
        target_size = len(all_predict_target)  # This will be 99
        specific_target = target

    print(f"Prediction targets: {predict_target}")
    print(f"Target size: {target_size}")

    # Image loading and preprocessing function (matching your training script)
    def load_and_preprocess_image(path):
        with rasterio.open(path) as src:
            bands = src.read()
            img = bands[:13]  # Use first 13 bands
            img = img / normalization  # Normalize to [0, 1]
        
        img = np.nan_to_num(img, nan=0, posinf=1, neginf=0)
        img = np.clip(img, 0, 1)  # Clip values to be within the 0-1 range
        img = np.transpose(img, (1, 2, 0))
        # Scale back to [0, 255] for visualization purposes
        img = (img * 255).astype(np.uint8)
        return img

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Swin transformer model (matching your training script)
    try:
        if model_name == 'swin_b':
            base_model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
        elif model_name == 'swin_s':
            base_model = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1)
        elif model_name == 'swin_t':
            base_model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Modify the first convolution layer to accept 13 input channels (matching your training)
        num_input_channels = 13
        base_model.features[0][0] = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=base_model.features[0][0].out_channels,
            kernel_size=base_model.features[0][0].kernel_size,
            stride=base_model.features[0][0].stride,
            padding=base_model.features[0][0].padding,
            bias=(base_model.features[0][0].bias is not None)
        )
        
        print(f"Successfully loaded {model_name} from torchvision")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise

    # Use your ViTForRegression wrapper (matching your training script)
    model = ViTForRegression(base_model, emb_size=emb_size, predict_size=target_size)

    # Load checkpoint if specified
    if use_checkpoint:
        if os.path.exists(checkpoint):
            print(f"Loading checkpoint: {checkpoint}")
            state_dict = torch.load(checkpoint, map_location=device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict["model_state_dict"])
            else:
                model.load_state_dict(state_dict)
            print("Checkpoint loaded successfully")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint}")
            print("Using pretrained model without fine-tuned weights")

    # Dataset class (matching your training script)
    class CustomDataset(Dataset):
        def __init__(self, dataframe, transform, predict_target):
            self.dataframe = dataframe
            self.transform = transform
            self.predict_target = predict_target

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            item = self.dataframe.iloc[idx]
            image = load_and_preprocess_image(item['imagery_path'])
            image_tensor = self.transform(image)
            
            # ALWAYS return all targets (99) to match model architecture
            target_values = []
            for col in self.predict_target:
                val = item[col]
                if pd.isna(val):
                    target_values.append(0.0)
                else:
                    try:
                        target_values.append(float(val))
                    except (ValueError, TypeError):
                        target_values.append(0.0)
            target = torch.tensor(target_values, dtype=torch.float32)
            
            return image_tensor, target

    # Image transforms (matching your training script exactly)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Resize((imagery_size, imagery_size)),  # Resize to match training
        transforms.Normalize(mean=[0.5] * 13, std=[0.22] * 13),  # Your exact normalization
    ])

    # Create datasets and dataloaders
    train_dataset = CustomDataset(train_df, transform, predict_target)
    val_dataset = CustomDataset(test_df, transform, predict_target)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    model.to(device)
    model.eval()

    print(f"Extracting features from {len(train_dataset)} training samples...")
    
    # Extract features from training data
    X_train, y_train = [], []
    for images, targets in tqdm(train_loader, desc="Processing training data"):
        images, targets = images.to(device), targets.to(device)
        with torch.no_grad():
            # Get features from the base model (before regression head)
            features = model.base_model.features(images)
            features = model.base_model.norm(features)
            features = model.base_model.permute(features)
            features = model.base_model.avgpool(features)
            features = torch.flatten(features, 1)
            
        X_train.append(features.cpu()[0].numpy())
        
        # Handle single vs multiple targets
        if targets.dim() == 0:
            y_train.append(targets.cpu().numpy().item())
        else:
            y_train.append(targets.cpu()[0].numpy())

    print(f"Extracting features from {len(val_dataset)} test samples...")
    
    # Extract features from test data
    X_test, y_test = [], []
    for images, targets in tqdm(val_loader, desc="Processing test data"):
        images, targets = images.to(device), targets.to(device)
        with torch.no_grad():
            # Get features from the base model (before regression head)
            features = model.base_model.features(images)
            features = model.base_model.norm(features)
            features = model.base_model.permute(features)
            features = model.base_model.avgpool(features)
            features = torch.flatten(features, 1)
            
        X_test.append(features.cpu()[0].numpy())
        
        # Handle single vs multiple targets
        if targets.dim() == 0:
            y_test.append(targets.cpu().numpy().item())
        else:
            y_test.append(targets.cpu()[0].numpy())

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Training targets shape: {y_train.shape}")
    print(f"Test targets shape: {y_test.shape}")


    # Save extracted features
    results_folder = f"results/swin_{mode}/{imagery_source}_{fold}/"
    os.makedirs(results_folder, exist_ok=True)
    
    pd.DataFrame(X_train).to_csv(results_folder + "X_train.csv", index=False)
    pd.DataFrame(y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train, 
                 columns=["target"] if y_train.ndim == 1 else [f"target_{i}" for i in range(y_train.shape[1])]).to_csv(
                 results_folder + "y_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(results_folder + "X_test.csv", index=False)
    pd.DataFrame(y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test,
                 columns=["target"] if y_test.ndim == 1 else [f"target_{i}" for i in range(y_test.shape[1])]).to_csv(
                 results_folder + "y_test.csv", index=False)

    print("Features saved to:", results_folder)

    # Ridge regression evaluation
    print("Running Ridge regression evaluation...")
    
    # Extract the specific target we want for evaluation
    if target != "":
        # Find the index of our target in the predict_target list
        if target in predict_target:
            target_idx = predict_target.index(target)
            y_train_eval = y_train[:, target_idx]
            y_test_eval = y_test[:, target_idx]
            print(f"Using target '{target}' at index {target_idx} for evaluation")
        else:
            # If target not found in predict_target, try to get it from original dataframe
            print(f"Target '{target}' not found in predict_target list")
            print(f"Available targets: {predict_target[:10]}...")  # Show first 10
            
            # Try to get from original dataframes
            if target in train_df.columns and target in test_df.columns:
                y_train_eval = train_df[target].values
                y_test_eval = test_df[target].values
                print(f"Using target '{target}' from original dataframe")
            else:
                print(f"Target '{target}' not found anywhere. Using first target.")
                y_train_eval = y_train[:, 0]
                y_test_eval = y_test[:, 0]
    else:
        # If no specific target, use the first one
        y_train_eval = y_train[:, 0]
        y_test_eval = y_test[:, 0]
        print("Using first target for evaluation (no specific target specified)")
    
    print(f"Evaluation target range - Train: [{y_train_eval.min():.3f}, {y_train_eval.max():.3f}]")
    print(f"Evaluation target range - Test: [{y_test_eval.min():.3f}, {y_test_eval.max():.3f}]")
    print(f"Evaluation target mean - Train: {y_train_eval.mean():.3f}, Test: {y_test_eval.mean():.3f}")



    alphas = np.logspace(-6, 6, 20)
    ridge_pipeline = Pipeline([
        ("ridge", RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_absolute_error"))
    ])

    # Cross-validation on training data
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        ridge_pipeline, X_train, y_train_eval, cv=kf, scoring="neg_mean_absolute_error"
    )

    print("Cross-validation scores (negative MAE):", cv_scores)
    print("Mean cross-validation score (negative MAE):", cv_scores.mean())

    # Final evaluation on test set
    ridge_pipeline.fit(X_train, y_train_eval)
    test_predictions = ridge_pipeline.predict(X_test)
    test_score = np.mean(np.abs(test_predictions - y_test_eval))
    
    print(f"Test Score (MAE): {test_score:.4f}")
    print(f"Test Score (negative MAE): -{test_score:.4f}")

    return test_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Swin transformer model evaluation for satellite imagery.')
    parser.add_argument('--fold', type=str, default='1', help='The fold number')
    parser.add_argument('--model_name', type=str, default='swin_b', 
                        help='The Swin model name (swin_b, swin_s, swin_t)')
    parser.add_argument('--target', type=str, default='', help='The target variable')
    parser.add_argument('--imagery_source', type=str, default='L', help='L for Landsat and S for Sentinel')
    parser.add_argument('--imagery_path', type=str, required=True, help='The parent directory of all imagery')
    parser.add_argument('--mode', type=str, default='spatial', choices=['temporal', 'spatial', 'one_country'],
                        help='Evaluation mode: temporal, spatial, or one_country')
    parser.add_argument('--model_output_dim', type=int, default=1024, 
                        help='The output dimension of the model')
    parser.add_argument('--emb_size', type=int, default=1000, 
                        help='Embedding size from training script')
    parser.add_argument('--use_checkpoint', action='store_true', 
                        help='Whether to use checkpoint file. If not, use raw pretrained model.')
    parser.add_argument('--model_not_named_target', action='store_false', 
                        help='Whether the model name contains the target variable')
    parser.add_argument('--grouped_bands', nargs='+', type=int, 
                        default=list(range(1, 14)),
                        help="List of grouped bands (default: 1-13 for all bands)")
    
    args = parser.parse_args()
    
    print(f"Starting Swin evaluation with model: {args.model_name}")
    print(f"Mode: {args.mode}, Imagery source: {args.imagery_source}")
    print(f"Using {len(args.grouped_bands)} bands")
    
    maes = []
    
    if args.mode == 'temporal':
        mae = evaluate_swin(
            "1", args.model_name, args.target, args.use_checkpoint,
            args.model_not_named_target, args.imagery_path, args.imagery_source, 
            args.mode, args.model_output_dim, args.grouped_bands, args.emb_size
        )
        print(f"Temporal evaluation MAE: {mae:.4f}")
        
    elif args.mode == 'spatial':
        print("Running 5-fold spatial cross-validation...")
        for i in range(5):
            fold = i + 1
            print(f"\n--- Fold {fold} ---")
            try:
                mae = evaluate_swin(
                    str(fold), args.model_name, args.target, args.use_checkpoint,
                    args.model_not_named_target, args.imagery_path, args.imagery_source,
                    args.mode, args.model_output_dim, args.grouped_bands, args.emb_size
                )
                maes.append(mae)
                print(f"Fold {fold} MAE: {mae:.4f}")
            except Exception as e:
                print(f"Error in fold {fold}: {e}")
        
        if maes:
            mean_mae = np.mean(maes)
            std_err = np.std(maes) / np.sqrt(len(maes))
            print(f"\n=== Spatial Cross-Validation Results ===")
            print(f"Mean MAE: {mean_mae:.4f} ± {std_err:.4f}")
            print(f"Individual fold MAEs: {maes}")
        
    elif args.mode == 'one_country':
        COUNTRIES = [
            'Madagascar', 'Burundi', 'Uganda', 'Mozambique', 'Rwanda',
            'Zambia', 'Tanzania', 'Malawi', 'Ethiopia', 'Kenya', 'Zimbabwe',
            'Lesotho', 'South Africa', 'Angola', 'Eswatini', 'Comoros'
        ]
        
        n_samples = len(COUNTRIES)
        print("Running one-country evaluation...")
        
        for country in COUNTRIES:
            print(f"\n--- {country} ---")
            try:
                mae = evaluate_swin(
                    country, args.model_name, args.target, args.use_checkpoint,
                    args.model_not_named_target, args.imagery_path, args.imagery_source,
                    args.mode, args.model_output_dim, args.grouped_bands, args.emb_size
                )
                maes.append(mae)
                print(f"{country} MAE: {mae:.4f}")
            except Exception as e:
                print(f"Error in {country}: {e}")
                n_samples -= 1
        
        if maes:
            mean_mae = np.mean(maes)
            std_err = np.std(maes) / np.sqrt(n_samples)
            print(f"\n=== One-Country Evaluation Results ===")
            print(f"Mean MAE: {mean_mae:.4f} ± {std_err:.4f}")
            print(f"Successful evaluations: {len(maes)}/{len(COUNTRIES)}")
    
    else:
        raise Exception(f"Invalid mode: {args.mode}")