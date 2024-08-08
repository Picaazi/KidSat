# KidSatExt
We have two aims with this extension of the KidSat project:
1. Improve the structure and commenting of some of the code from the KidSat project
2. Use the DinoV2 model to predict orphanhood (as a proportion, and later as a count) in Zambia in 2018, 2020 and 2023. With the aim to compare this to data 
we have in the Sinazongwe District.

If you are viewing this from the KidSat project, these are the main changes that I have made:
1. Changed the joins of the KR, IR and PR, and tidied up the code of ```survey_processing/main.py``` so that children from 6 to 18 are now included in the training data for the models.
2. Tidied up, restructured and commented several files. As one example, I've commented and restructured ```evaluate_orphanhood.py```, which is the counterpart to ```evaluate.py``` from the KidSat project. This file only needs a couple of small changes to be used for predicting child deprivation for the KidSat project.
3. Added a more in-depth set of instructions for getting all the data, setting up the google cloud compute engine, training the model and getting predictions and orphanhood maps.

Here is an overall description of how we plan to predict orphanhood:
1. Get DHS data, use this to create our child deprivation indicators (we call poverty variables).
2. Aggregate DHS data, poverty variables to the cluster level and combine with GPS data.
3. Associate a satellite image with each cluster.
4. Finetune our DinoV2 model on the satellite imagery to predict [proportion of people who have lost a mother, ... lost a father], or this vector + the 99 dimension child deprivation vector from the KidSat project.
5. Then we add a ridge regression layer to our DinoV2 model that outputs one value, orphanhood. We fit this regression layer with the satellite imagery and orphanhood data.
6. Now we can freely evaluate our model on a grid of satellite imagery covering a whole country, say Zambia and display a chorolopleth map of orphanhood.

## Instructions

### Intial Setup

The data step is suitably quick to run on your local computer, or this can all be done on a VM on GCP. Create a virtual environment and install all the modules in ```requirements.txt```. Note that the code is currently configured to predict orphanhood in Zambia for under 16s.

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
python survey_processing.py config_options_i_plan_to_change
```
The resulting training and test data for our models will be stored in ```survey_processing/processed_data```.

### Satellite Imagery

We now need to download the satellite imagery at each of the clusters in the DHS data. For this project we have typically used 10km x 10km images, this is partially due to the jitter of the DHS data. If you are lucky, someone will have done this for you, i.e safely stored on the MLGH Google Drive. Otherwise you will need to extract the coordinates for each of the clusters using ```geopandas``` on the geographic Shape files. These coordinates will need to be stored in a ```DataFrame``` with columns ```name, lat, lon``` where ```name``` is the cluster ID. 

To download these satellite images you will need to code a very short script utilising ```imagery_scraping/download_imagery.py```. Firstly, update the GEE project name in the config file ```imagery_scraping/config/google_config.json```. Then you only need to load the ```DataFrame``` mentioned above for each survey, and call the ```download_imagery()``` function from ```download_imagery.py```. GEE caps the number of requests to 3000 at a time, so you will need to run the script repeatedly. You may wish to use the python ```OS``` module to count the files you have download to check none are missed. It is recommended to store these satellite images at ```KidSatExt/imagery```, but this is not required.

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
1. Python, git etc should be already installed so begin by cloning KidSatExt.
2. The Deep Learning VM uses a conda virutal environment, install the modules from ```requirements.txt```.
3. Copy the data in ```processed_data``` from the Cloud Bucket to the VM using ```gcloud storage```.
4. To load the imagery when training the dino model, we need each images file path. The images can either be copied to the VM from the Bucket, or to a disk drive that can be attached to the VM. It is possible to load the images directly from the Cloud Bucket, this will require some adaptations to the code in ```modelling/dino```.

### Dino Model Training

DinoV2 is a model that can be used for a range of computer vision tasks. It is created by Facebook and trained on millions of images. It can take varying size images as an input, although ideally all images should be the same size.  We can finetune this model on additional images.

The model is trained in two stages. First we finetune the Dino model alone. The input to our model is a satellite image for a cluster. And the target data is either [the proportion of children who have lost a mother, ... lost a father], or this vector + the 99 dimension child poverty vector from the KidSat project. Then we add a ridge regression layer to the end of our Dino model, which will only output 1 value, orphanhood. This ridge regression layer is trained using the satellite imagery and the proportion of orphans in each cluster. One model is trained on each fold, and we will pick the best model out of the 5 at then end.

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

We can now choose the best trained Dino model. We can use ```modelling/dino/predict_orphanhood.py``` to get orphanhood predictions in the form of a ```DataFrame``` with columns ```lat, lon, orphaned```. To predict orphanhood for a certain country, we need to download more satellite imagery, covering the whole country. Follow all the previous steps to do this, but it is recommend to store the images at ```prediction_data/imagery_folder_name```. Then run the following command:
```
python modelling/dino/predict_orphanhood.py various_command_line_options
```
These predictions are stored at ```prediction_data/orphanhood_predictions.csv```.

We can then run the Python Notebook ```create_choropleth_map.ipynb``` to get a choropleth map of orphanhood. This file is currently configured to create an orphanhood map for Zambia, but other maps can be made by downloading the appropriate map file from ```https://gadm.org/```.

