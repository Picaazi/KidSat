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


def prepare_location_features(df):
    """
    Prepare location features from DHS data (already processed)
    """
    # hv025 is already one-hot encoded as hv025_1 and hv025_2
    location_columns = ['hv025_1', 'hv025_2']
    
    # Check if columns exist
    missing_cols = [col for col in location_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing location columns: {missing_cols}")
    
    # Extract urban/rural features
    urban_rural_features = df[location_columns].values
    
    print(f"Urban/Rural feature distribution:")
    print(f"  hv025_1 (Urban): {df['hv025_1'].sum()} samples ({df['hv025_1'].mean()*100:.1f}%)")
    print(f"  hv025_2 (Rural): {df['hv025_2'].sum()} samples ({df['hv025_2'].mean()*100:.1f}%)")
    
    # Optional: Add additional location features if you want to experiment
    additional_features = []
    
    # Combine all location features
    if additional_features:
        all_features = np.concatenate([urban_rural_features] + additional_features, axis=1)
        total_additional = sum(f.shape[1] for f in additional_features)
        print(f"  Total location features: {all_features.shape[1]} (2 urban/rural + {total_additional} additional)")
    else:
        all_features = urban_rural_features
        print(f"  Total location features: {all_features.shape[1]} (urban/rural only)")
    
    return all_features


def evaluate(
    fold,
    model_name,
    target="",
    use_checkpoint=False,
    model_not_named_target=True,
    imagery_path=None,
    imagery_source=None,
    mode="temporal",
    model_output_dim=768,
    grouped_bands=None,
    country=None,
    enhanced_targets=False,
    use_location_features=False, # Use Geo info 
):
    model_par_dir = "modelling/dino/model/"
    country_suffix = f'_{country.upper()}' if country else ''
    enhanced_suffix = f'_enhanced' if enhanced_targets else ''

    # Build checkpoint filename (pth file) based on mode and target
    if use_checkpoint:
        named_target = target if model_not_named_target else ""
        if mode == "temporal":
            checkpoint = f"{model_par_dir}{model_name}_temporal_best_{imagery_source}{named_target}{country_suffix}{enhanced_suffix}.pth"
        elif mode == "spatial":
            checkpoint = f"{model_par_dir}{model_name}_{fold}_{grouped_bands}all_cluster_best_{imagery_source}{named_target}{country_suffix}{enhanced_suffix}.pth"
        elif mode == "one_country":
            checkpoint = f"{model_par_dir}{model_name}_{fold}_one_country_best_{imagery_source}{named_target}{country_suffix}{enhanced_suffix}.pth"
        else:
            raise Exception(mode)

    print(
        f"Evaluating {model_name} on fold {fold} {f'for country {country_suffix[1:]}' if country else '' } with target {target} using checkpoint {checkpoint if use_checkpoint else 'None'}"
    )

    # Modified to adjust the actual number of features/column (target size) of the country-wise model
    if use_checkpoint and os.path.exists(checkpoint):
        # Load checkpoint to get the actual target size
        temp_state = torch.load(checkpoint, map_location='cpu')
        actual_target_size = temp_state['model_state_dict']['regression_head.weight'].shape[0]
        target_size = actual_target_size
        print(f"Detected target size from checkpoint: {target_size}")
        
        # Also set eval_target appropriately
        if target == "":
            eval_target = "deprived_sev"  # Use this for single target evaluation
    else:
        # Original logic for when not using checkpoint
        if target == "":
            eval_target = "deprived_sev"
            if enhanced_targets:
                target_size = 101
            else:
                target_size = 99
        else:
            eval_target = target
            target_size = 1 if model_not_named_target else 99

    # Determine size of target
    '''
    if target == "":
        eval_target = "deprived_sev"
        target_size = 99
    else:
        eval_target = target
        target_size = 1 if model_not_named_target else 99
    '''

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
        train_df = pd.read_csv(f"{data_folder}train_fold_{fold}{country_suffix}.csv")
        test_df = pd.read_csv(f"{data_folder}test_fold_{fold}{country_suffix}.csv")

    
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

    # Reset indices after all filtering is complete
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # NOW extract location features from the final, clean dataframes
    if use_location_features:
        print("Preparing training location features...")
        train_location_features = prepare_location_features(train_df)
        
        print("Preparing test location features...")
        test_location_features = prepare_location_features(test_df)
    else:
        train_location_features = None
        test_location_features = None
        print("Not using location features")

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
    X_train_visual, y_train = [], []

    for images, targets in tqdm(train_loader):
        images, targets = images.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model.base_model(images)
        X_train_visual.append(outputs.cpu()[0].numpy())
        y_train.append(targets.cpu()[0].numpy())

    # Extract features from base model for test data
    X_test_visual, y_test = [], []
    for images, targets in tqdm(val_loader):
        images, targets = images.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model.base_model(images)
        X_test_visual.append(outputs.cpu()[0].numpy())
        y_test.append(targets.cpu()[0].numpy())

    X_train_visual, y_train = np.array(X_train_visual), np.array(y_train)
    X_test_visual, y_test = np.array(X_test_visual), np.array(y_test)

    # Combine visual and location features
    if use_location_features:
        # Since DataLoader processes samples in order and we reset indices,
        # location features align directly with visual features
        X_train = np.concatenate([X_train_visual, train_location_features], axis=1)
        X_test = np.concatenate([X_test_visual, test_location_features], axis=1)
        
        print(f"\nCombined feature dimensions:")
        print(f"  Visual features: {X_train_visual.shape[1]}")
        print(f"  Location features: {train_location_features.shape[1]}")
        print(f"  Total features: {X_train.shape[1]}")
    else:
        X_train = X_train_visual
        X_test = X_test_visual
        print(f"\nUsing visual features only: {X_train.shape[1]} dimensions")
    

    # Save extracted features and targets to CSV
    results_folder = (
        f"modelling/dino/results/split_{mode}{imagery_source}_{fold}_{grouped_bands}"
        f"{'_loc' if use_location_features else ''}{country_suffix}"
    )
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    pd.DataFrame(X_train).to_csv(results_folder + "X_train.csv", index=False)
    pd.DataFrame(y_train, columns=["target"]).to_csv(results_folder + "y_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(results_folder + "X_test.csv", index=False)
    pd.DataFrame(y_test, columns=["target"]).to_csv(results_folder + "y_test.csv", index=False)

    # Save visual features separately for comparison
    pd.DataFrame(X_train_visual).to_csv(results_folder + "X_train_visual_only.csv", index=False)
    pd.DataFrame(X_test_visual).to_csv(results_folder + "X_test_visual_only.csv", index=False)

    # Save location features if used
    if use_location_features:
        pd.DataFrame(train_location_features).to_csv(results_folder + "X_train_location.csv", index=False)
        pd.DataFrame(test_location_features).to_csv(results_folder + "X_test_location.csv", index=False)

    # Ridge Regression with cross-validation to evaluate features
    alphas = np.logspace(-6, 6, 20)

    # Define the pipeline once
    ridge_pipeline = Pipeline([
        ("ridge", RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_absolute_error"))
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # COMPARATIVE ANALYSIS
    if use_location_features:
        print("\n=== COMPARATIVE ANALYSIS ===")
        
        # Visual features only
        visual_cv_scores = cross_val_score(
            ridge_pipeline, X_train_visual, y_train, cv=kf, scoring="neg_mean_absolute_error"
        )
        ridge_pipeline.fit(X_train_visual, y_train)
        visual_only_score = np.mean(np.abs(ridge_pipeline.predict(X_test_visual) - y_test))
        
        print(f"Visual features only:")
        print(f"  CV MAE: {-visual_cv_scores.mean():.4f} ± {visual_cv_scores.std():.4f}")
        print(f"  Test MAE: {visual_only_score:.4f}")
        
        # Location features only (if meaningful)
        if train_location_features.shape[1] > 0:
            location_cv_scores = cross_val_score(
                ridge_pipeline, train_location_features, y_train, cv=kf, scoring="neg_mean_absolute_error"
            )
            ridge_pipeline.fit(train_location_features, y_train)
            location_only_score = np.mean(np.abs(ridge_pipeline.predict(test_location_features) - y_test))
            
            print(f"Location features only:")
            print(f"  CV MAE: {-location_cv_scores.mean():.4f} ± {location_cv_scores.std():.4f}")
            print(f"  Test MAE: {location_only_score:.4f}")
        else:
            location_only_score = None
        
        # Combined features
        combined_cv_scores = cross_val_score(
            ridge_pipeline, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error"
        )
        ridge_pipeline.fit(X_train, y_train)
        combined_score = np.mean(np.abs(ridge_pipeline.predict(X_test) - y_test))
        
        print(f"Combined features:")
        print(f"  CV MAE: {-combined_cv_scores.mean():.4f} ± {combined_cv_scores.std():.4f}")
        print(f"  Test MAE: {combined_score:.4f}")
        
        # Calculate improvement
        improvement = visual_only_score - combined_score
        improvement_pct = (improvement / visual_only_score) * 100
        print(f"\nImprovement from adding location: {improvement:.4f} MAE ({improvement_pct:.1f}%)")
        
        # Save detailed analysis
        analysis_results = {
            'visual_only_cv_mae': -visual_cv_scores.mean(),
            'visual_only_cv_std': visual_cv_scores.std(),
            'visual_only_test_mae': visual_only_score,
            'location_only_test_mae': location_only_score,
            'combined_cv_mae': -combined_cv_scores.mean(),
            'combined_cv_std': combined_cv_scores.std(),
            'combined_test_mae': combined_score,
            'improvement_mae': improvement,
            'improvement_pct': improvement_pct,
            'visual_features_count': X_train_visual.shape[1],
            'location_features_count': train_location_features.shape[1],
            'total_features_count': X_train.shape[1]
        }
        
        pd.DataFrame([analysis_results]).to_csv(results_folder + "feature_analysis.csv", index=False)
        
        # Use combined results for final reporting
        final_cv_scores = combined_cv_scores
        final_test_score = combined_score
    
    else:
        # Standard evaluation without location features
        final_cv_scores = cross_val_score(
            ridge_pipeline, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error"
        )
        ridge_pipeline.fit(X_train, y_train)
        final_test_score = np.mean(np.abs(ridge_pipeline.predict(X_test) - y_test))

    # Final results
    print(f"\n=== FINAL EVALUATION RESULTS ===")
    print("Cross-validation scores (negative MAE):", final_cv_scores)
    print("Mean cross-validation score (negative MAE):", final_cv_scores.mean())
    print("Test Score (MAE):", final_test_score)

    return final_test_score



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
    parser.add_argument('--country', type=str, help='Two-letter country code for single country training (e.g., ET, KE)')
    parser.add_argument('--use_location_features', action='store_true', help='Whether to include urban/rural and other location features')
    parser.add_argument('--enhanced_targets', action='store_true')

    args = parser.parse_args()
    maes = []
    if args.mode == 'temporal':
        print(evaluate("1", args.model_name,args.target, args.use_checkpoint,args.model_not_named_target, args.imagery_path, args.imagery_source, args.mode,  args.model_output_dim))
    elif args.mode == 'spatial':
        for i in range(5):
            fold = i + 1
            mae = evaluate(str(fold), args.model_name, args.target, args.use_checkpoint,args.model_not_named_target,args.imagery_path, args.imagery_source, args.mode, args.model_output_dim, grouped_bands=args.grouped_bands, country=args.country, enhanced_targets=args.enhanced_targets, use_location_features=args.use_location_features)
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