<<<<<<< HEAD
# KidSat: satellite imagery to map childhood poverty

## Introduction

This is a repository for the work **KidSat: satellite imagery to map childhood poverty**.

![Figure 1](https://i.imgur.com/xLbiFwq.png)


## Getting All DHS Data

The Demographic and Health Surveys (DHS) program gathers and shares vital data on population, health, and nutrition in developing countries to inform public health policies. Their collection procedures and methods are listed [here](https://dhsprogram.com/data/data-collection.cfm).

To access DHS data, please follow these steps:

1. **Register for DHS Access:**
   - Visit the registration page [here](https://dhsprogram.com/data/new-user-registration.cfm) and apply for access to the DHS data.


2. **Obtain the Data for Following Countries and Years**
    For the following country and years, select ALL STATA and Geographic Data.
    | Country      | Year(s) |
    |--------------|---------|
    | Zambia       | 2007, 2013, 2018|
    | Malawi       | 2000, 2004, 2010, 2015|
    | Uganda       | 2000, 2006, 2011, 2016|
    | Comoros      | 2012|
    | Tanzania     | 1999, 2010, 2015, 2022|
    | Kenya        | 2003, 2008, 2014, 2022|
    | Angola       | 2015    |
    | Ethiopia     | 2000, 2005, 2011, 2016, 2019|
    | Rwanda       | 2005, 2007, 2010, 2014, 2019|
    | Lesotho      | 2004, 2009, 2014    |
    | Madagascar   | 1997, 2008, 2021|
    | Zimbabwe     | 1999, 2005, 2010, 2015|
    | Burundi      | 2010, 2016    |
    | Mozambique   | 2011    |
    | Eswatini     | 2006    |
    | South Africa | 2016    |

    The folders should be unzipped and store in `survey_processing/dhs_data/` (e.g. `survey_processing/dhs_data/` should contain subfolders of "ET_20XX_DHS_XXX..." etc. ).
---

## Usage for Imagery Scraping

This section provides step-by-step instructions on how to use this repository to achieve its intended functionality.

### Prerequisites

Before you start, make sure you have registered a Google Earth Engine project for academic purposes. You will need your project name to query the API. The sign-up page is [here](https://signup.earthengine.google.com).


1. **Set Up Environment**

    Example:

    ```bash
    pip install -r requirements.txt
    ```

2. **Configuration**

    You need to update your Google Earth Engine project name to `imagery_scraping/config/google_config.json`. The format (for me) was `ee-YOUR_GMAIL_NAME`. Note, please do not push your project name to GitHub.

3. **Query File (Optional)**

    The file `imagery_scraping/config/query.json` contains an example of how you should query imageries. You need to provide the latitude and longitude in WGS84 format. In our work, we mainly use shapefile from DHS directly.

4. **Running the Application**
    You first need to go to the `imagery_scraping` directory

    Example:


    An example of usage is shown below:

    ```bash
    python main.py "config/query.csv" "EarthImagery" 2021 "L8" -r 5
    ```

    It will prompt you to authenticate for Google. If all goes well, it will download the images to your Google Drive under a folder called `EarthImagery`. The images will be collected from the 2021 LandSat8 dataset and will be centered around the coordinates you provided in the query file with a 5 km square window.

    If you have a shapefile from DHS, you can also use for example

    ```bash
    python main.py "ETGE81FL" "Ethiopia2021Imagery" 2021 "S2" -r 5
    ```
    
    to extract the imagery.

5. **Visualization (Optional)**

    To see the imagery, you need to download the imagery data from Google Drive first. We provide sample data in `imagery_scraping/data` and a [notebook](imagery_scraping/visualization.ipynb) to see the imagery you queried in true color. Note that this is only a visualization; the original data is much richer and contains more than the three RGB channels. For training, we should use the original data instead of the true-color image alone.

6. **Getting All Imagery**

    We recommend using this [notebook](imagery_scraping/get_imagery.ipynb) to download all imagery and keep track of progress as GEE has a upper limit of 3000 jobs at the same time. You will need to download the imagery and save to an accessible location (we will refer to `path_to_parent_imagery_folder` in later sections), each of its subdirectory should be country code + year + source (e.g. ET2019S2 for Ethiopia 2019 Sentinel 2). The notebook should already be formatting the export using this naming convention.


## Summarizing the Dataset

Collect all DHS data to `survey_processing/dhs_data`. The following command

```bash
python survey_processing/main.py survey_processing/dhs_data
```

would create 5 splits of the training and test data for spatial analysis and before/after 2020 split for temporal analysis.

## Experiment with MOSAIKS

The MOSAIKS features were extracted using [IDinsight](https://github.com/IDinsight/mosaiks#mosaiks-satellite-imagery-featurization) package. A [notebook](modelling/mosaiks/main.ipynb) is provided in this repository for getting all features for MOSAIKS.

## Experiment with DINOv2

After having the splits in `survey_processing/processed_data`, you can finetune DINOv2 using the following commands. For the spatial experiment with Landsat imagery, you can use the following code.


```bash
python modelling/dino/finetune_spatial.py --fold 1 --model_name dinov2_vitb14 --imagery_path {path_to_parent_imagery_folder} --batch_size 8 --imagery_source L --num_epochs 20
```

Finetuning sentinel imagery, the normal command is 

```bash
python modelling/dino/finetune_spatial.py --fold 1 --model_name dinov2_vitb14 --imagery_path {path_to_parent_imagery_folder} --batch_size 1 --imagery_source S --num_epochs 10
```

Note that to get a cross-validated result, you should use fold 1 to 5.

For temporal finetuning, the command for Landsat is 

```bash
python modelling/dino/finetune_temporal.py --model_name dinov2_vitb14 --imagery_path {path_to_parent_imagery_folder} --batch_size 8 --imagery_source L
```

and replace `L` to `S` for sentinel finetuning.

For evaluation, make sure the all 1-5 finetuned spatial models  (or the finetuned temporal model for temporal evaluation) are in `modelling/dino/model` and run 

```bash
python modelling/dino/evaluate.py --use_checkpoint --imagery_path {path_to_parent_imagery_folder} --imagery_source L --mode spatial
```

Change the `--mode` to `temporal` for temporal evaluation, and change `L` to `S` for imagery sources.
Remove the `--use_checkpoint` for evaluating on raw DINO models.

## Experiment with SatMAE
### Finetuning
To run the finetuning process, you first need to download the checkpoints for fMoW-SatMAE [non-temporal](https://zenodo.org/record/7369797/files/fmow_pretrain.pth) or [temporal](https://zenodo.org/record/7369797/files/pretrain_fmow_temporal.pth). Then run the following:

```sh
python -m modelling.satmae.satmae_finetune --pretrained_ckpt $CHECKPOINT_PATH --dhs_path ./survey_processing/processed_data/train_fold_1.csv --output_path $OUTPUT_DIR --imagery_path $IMAGERY_PATH
```
Arguments:
- `--pretrained_ckpt`: Checkpoint of pretrained SatMAE model.
- `--imagery_path`: Path to imagery folder
- `--dhs_path`: Path to DHS `.csv` file
- `--output_path`: Path to export the output. A unique subdirectory will be created.
- `--batch_size`
- `--random_seed`
- `--sentinel`: Landsat is used by default. Turn this on to use Sentinel imagery
- `--temporal`: Add this flag to use the temporal mode
- `--epochs`: Number of epochs
- `--stopping_delta`: Delta for early stopping
- `--stopping_patience`: Early stopping patience
- `--loss`: Either `l1` (default) or `l2`.
- `--lr`: Learning rate
- `--weight_decay`: Weight decay for Adam optimizer
- `--enable_profiling`: Enable reporting of loading/inference time.


### Evaluation
Evaluation consists of 2 steps: exporting the model output, and perform Ridge Regression. Since exporting the model output is expensive, we split it into 2 separate modules:

To carry out the first step, edit the file `modelling/satmae/satmae_eval` and change the `SATMAE_PATHS` variable accordingly. For each entry, you can put all the model checkpoints you need to evaluate or `None` to use the pretrained checkpoint, along with their fold (1-5). You do not have to put the entries in any order, nor need to put all the folds, but the script caches the data from different folds in memory, which helps significantly reduce the time for loading and preprocessing the satellite images.
```sh
python -m modelling.satmae.satmae_eval --output_path $OUTPUT_DIR --imagery_path $IMAGERY_PATH
```
Arguments
- `--imagery_path`: Path to imagery folder
- `--output_path`: Path to export the output. A unique subdirectory will be created.
- `--batch_size`
- `--sentinel`: Landsat is used by default. Turn this on to use Sentinel imagery
- `--temporal`: Add this flag to use the temporal mode

This will export data as Numpy arrays in `.npy` files in the output location, which has the shape `(num_samples, 1025)`. The first 1024 columns (i.e `arr[:, :1024]`) is the predicted feature vector from the model, and the last column (i.e `arr[:, 1024]`) is the target. You can then adapt the script `modelling/satmae/eval_dhs.py` to conduct Ridge Regression or more advanced regression.
=======
# KidSatExt
We have two aims with this extension of the KidSat project:
1. Use the DinoV2 model to predict orphanhood (as a proportion, and later as a count) in Zambia in 2018 and 2023. With the aim to compare this to data 
we have in the Sinazongwe District.
2. Review the code from the KidSat project and try to improve the structure and commenting.

If you are viewing this from the KidSat project, these are the main changes that I have made:
1. Changed the joins of the KR, IR and PR from ```survey_processing/main.py``` so that children from 6 to 18 are now included in the training data for the models. Redeisgned this code to make it is easier to understand and make changes to.
2. Tidied up, restructured and commented several files. As one example, I've commented and restructured ```evaluate_orphanhood.py```, which is the counterpart to ```evaluate.py``` from the KidSat project. This file only needs a couple of small changes to be used for predicting child deprivation for the KidSat project.
3. Added code to predict orphanhood given a collection of satellite images and center coordinates.
4. Added a more in-depth set of instructions for getting all the data, setting up the google cloud compute engine, training the model, getting predictions and orphanhood maps.

More detailed findings are found in the next subsection!

Here is an overall description of how we plan to predict orphanhood:
1. Get DHS data, use this to create our child deprivation indicators (we call poverty variables).
2. Aggregate DHS data, poverty variables to the cluster level and combine with GPS data.
3. Associate a satellite image with each cluster.
4. Finetune our DinoV2 model on the satellite imagery to predict the proportion of children who have lost a mother and the proportion of children who have lost a father, or this vector combined with the 99 dimension child deprivation vector from the KidSat project.
5. Then we add a ridge regression layer to our DinoV2 model that outputs one value, orphanhood. We fit this regression layer with the satellite imagery and orphanhood data.
6. Now we can freely evaluate our model on a grid of satellite imagery covering a whole country, say Zambia and display a chorolopleth map of orphanhood.

If you wish to move straight on to the setup instructions, skip this next section.

### My Findings and Suggestions for the KidSat Paper

1. Firstly I have made some big changes to the code of ```survey_processing/main.py```. In particular, I have changed the join of the PR and IR to the KR (under 5's dataset), from a left join to an outer join. As a consequence the data now includes 6 - 18 year olds. This means that there is a lot more data so the predictions from the KidSat paper could potentially be improved. Also, the 99 dimension vector from the KidSat paper mainly depends on data from the PR, and only a few variables from the KR. So these new 6 - 18 year olds will have enough data to contribute towards the 99 dimension vector.
2. In ```finetune.py``` and ```evaluate.py```, clusters that are missing any variable from the 99 dimension vector are removed. This means that the total number of clusters is reduced by a fair amount and all pre-2005 surveys are removed from the training data. There is also some data removed in ```survey_processing/main.py```. It may be worth checking if it would make sense to keep more of the data.
4. I suggest outputting the MAPE, MSE, MAE and R2 score everytime the model is trained since MAE does not give a full picture, due to the data being in the range [0, 1].
5. The file ```download_imagery.py``` gives the option to get imagery with all colour bands or RBG only. The RGB bands from these images however need to be read differently from each other. I have adapted ```load_and_preprocess_image()``` in ```predict_orphanhood.py``` to read the image bands in both all colour band and RGB only imagery.

## Instructions

### Intial Setup and Important Notes

The data step is suitably quick to run on your local computer, or this can all be done on a VM on GCP. Create a virtual environment and install all the modules in ```requirements.txt```. 

Note that ```predict_orphanhood.py``` needs some slight changes to work with Landsat imagery and the temporal Dino model.

### DHS data
First register for access to the DHS data in the necessary countries. For each country and year download all the Stata files, alongside the Geographic data (Shape file). This must be done manually, not via the bulk download manager. Store this data at ```survey_processing/dhs_data```. The file structure should be as follows:
```
dhs_data
  AO_2015_DHS_XXX...
    AOPR71DT
    AOIR71DT
    ...
  ET_2005_DHS_XXX...
    ...
  ...
```
Now in order to create the poverty variables, aggregate the data to the cluster level, split the data into 5 folds and into a pre/post 2020 fold, we need to run ```survey_processing/main.py``` by the following command:
```
python survey_processing.py --dhs_data_dir {path_to_dhs_data}
```
The resulting training and test data for our models will be stored in ```survey_processing/processed_data```.

### Satellite Imagery

We now need to download the satellite imagery at each of the clusters in the DHS data. For this project we have typically used 10km x 10km images, this is partially due to the jitter of the DHS data. If you are lucky, someone will have done this for you, i.e safely stored on the MLGH Google Drive. Otherwise you will need to extract the coordinates for each of the clusters using ```geopandas``` on the geographic Shape files. These coordinates will need to be stored in a ```DataFrame``` with columns ```name, lat, lon``` where ```name``` is the cluster ID (i.e ZM201800000023). 

To download these satellite images you will need to code a very short script utilising ```imagery_scraping/download_imagery.py```. Firstly, update the GEE project name in the config file ```imagery_scraping/config/google_config.json```. Then you only need to load the ```DataFrame``` mentioned above for each survey, and call the ```download_imagery()``` function from ```download_imagery.py```. GEE caps the number of requests to 3000 at a time, so you will need to run the script repeatedly. You may wish to use the python ```OS``` module to count the files you have downloaded to check none are missed. It is recommended to store these satellite images in a folder called ```imagery```.

### Google Cloud

To train our Dino model, it is necessary to utilise Google Cloud's Compute Engine. To create this VM and transfer our data to the cloud, follow these steps:
1. Create a project in GCP. If you are not the owner, grant yourself the appropriate IAM permissions.
1. Go to Compute Engine, select Create VM Instance.
2. Select any region, I have personally found Asia-SouthEast to have the most available GPUs.
3. On machine configuration, select GPU, then A100 40GB.
4. Increase the size of the boot disk and change the OS to Deep Learning VM with CUDA 11.8 M124 (Debian 11, Python 3.10). The important thing is making sure CUDA, torch and torchvision's versions are compatible.
5. Either allow full access to all Cloud API's or manually allow them after this setup.
6. Under Cloud Storage, create a Cloud Bucket, this is where we will upload our imagery and training/test data. Make sure it is created in the same region as the VM.
7. Upload to this by downloading any data to your local computer, then selecting Upload on the Cloud Bucket.
8. Alternatively, if the data is stored in a Google Drive you can using ```gcloud storage``` or ```gsutil``` to copy the files to the Bucket.
9. This can be done in Google Collab by mounting the Google Drive, and running the commands ```gcloud init```, ```gcloud storage ls``` and ```gcloud storage cp -r drive_folder cloud_bucket```.

Now follow these instructions to setup the VM from the command line:
1. Python, git etc should be already installed so begin by cloning the KidSatExt repository.
2. The Deep Learning VM uses a conda virutal environment, install the modules from ```requirements.txt```.
3. Copy the data in ```survey_processing/processed_data``` from the Cloud Bucket to the VM using ```gcloud storage```.
4. To load the imagery when training the dino model, we need each images file path. We can do this by mounting the Cloud Bucket to the VM using ```gcsfuse```. Create a directory for these images called ```dhs_imagery```.

### Dino Model Training

DinoV2 is a model that can be used for a range of computer vision tasks. It is created by Facebook and trained on millions of images. It can take varying size images as an input, although ideally all images should be the same size.  We can finetune this model on additional images.

The model is trained in two stages. First we finetune the Dino model alone. The input to our model is a satellite image for a cluster. And the target data is either the proportion of children who have lost a mother and the proportion of children who have lost a father, or it's this vector combined with the 99 dimension child poverty vector from the KidSat project. Then we add a ridge regression layer to the end of our Dino model, which will only output 1 value, orphanhood. This ridge regression layer is trained using the satellite imagery and the proportion of orphans in each cluster. One model is trained on each fold.

To finetune the dino model we run the following command for all 5 of the folds:
```
python modelling/dino/finetune_spatial_orphanhood.py --fold 1 --model_name dinov2_vitb14 --imagery_path {path_to_parent_imagery_folder} --batch_size 1 --imagery_source S --num_epochs 10
```

To then fit the ridge regression layer, and output the error metrics on each of the folds, we run this command once:
```
python modelling/dino/evaluate_orphanhood.py --use_checkpoint --imagery_path {path_to_parent_imagery_folder} --imagery_source S --mode spatial
```

The model's learned parameters, as well as the ridge regression parameters are stored at ```modelling/dino/model```. 

### Next Steps

We can now get some predictions. We can use ```modelling/dino/predict_orphanhood.py``` to output orphanhood predictions in the form of a ```DataFrame``` with columns ```name, lat, lon, orphaned, in_sample```. Where ```name``` is the centroid ID and ```in_sample``` is an indicator variable of whether the image has been used to train the model. Note this currently only works for the spatial Dino model trained on Sentinel imagery. This can be quite easily adapted.

As an example, to predict orphanhood for a certain country, we need to download more satellite imagery, covering the whole country. Create a folder called ```prediction_data``` and follow all the previous steps to download the new imagery and store it there. In this folder, we also want to store the image file names with their center coordinates in the form of a ```DataFrame``` with columns ```name, lat, lon```. Then run the following command:
```
python modelling/dino/predict_orphanhood.py --use_checkpoint --imagery_source S imagery_path {path_to_parent_imagery_folder} --data_path {path_to_imagery_coords_csv}
```
The predictions from each of the five models are then saved as ```prediction_data/orphanhood_predictions_fold_i.csv```.

We can then plot the true values vs the predictions if available or run the Python Notebook ```create_choropleth_map.ipynb``` to get a choropleth map of orphanhood. This file is currently configured to create an orphanhood map for Zambia, but other maps can be made by downloading the appropriate map file from ```https://gadm.org/```.

## My Orphanhood Prediction Results

I trained my orphanhood model using all the clusters in Sub-Saharan Africa (the countries from the KidSat Project). I used RGB Sentinel imagery. My Dino model first predicted the proportion of people who lost a mother and the proportion of people who lost a father. Then the ridge regression layer predicted orphanhood. I trained the model using the same hyperparameters as in the KidSat Project. A batch size of one and ten epochs. I used my adapted version of ```survey_processing/main.py``` to get the DHS and poverty data. This allowed me to make use of children of all ages in the data. My first set of results however did not turn out as planned: 

<div align="center">
  <img src="./orphanhood_predictions_out_sample.PNG" alt="Plot">
</div>

<div align="center">
  <img src="./orphanhood_predictions_in_sample.PNG" alt="Plot">
</div>

The performance on the best fold was as follows:
- MAE: 0.00423
- MSE: 0.05094
- MAPE: 35 trillion
- R2: 0.09224

Clearly the model is not effective at predicting orphanhood from a satellite image. I believe we need to change the first stage of the model so it predicts more than just proportion of people who lost a mother and lost a father. The deprivation should be an important covariate to orphanhood proportion and the results from the KidSat paper suggest their Dino model is reasonably effective at predicting the level of deprivation from satellite imagery. Hence I believe the model performance can be drastically improved if we first predict proportion of people who lost a mother and lost a father alongside the 99 dimension child deprivation vector, then predict orphanhood proportion. Unfortunately I ran out of time this summer to work on this. I'm hopeful that someone will be able to get an improved model.
>>>>>>> joshua_testing
