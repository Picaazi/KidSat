# KidSatExt
We have two aims with this extension of the KidSat project:
1. Improve the structure and commenting of some of the code from the KidSat project
2. Use the DinoV2 model to predict orphanhood (as a proportion, and later as a count) in Zambia in 2018, 2020 and 2023. With the aim to compare this to data 
we have in the Sinazongwe District.

If you are viewing this from the KidSat project, these are the main changes that I have made:
1. Changed the joins of the KR, IR and PR, and tidied up the code of survey_processing/main.py so that children from 6 to 18 are now included in the training data for the models.
2. Tidied up, restructured and commented several files. As one example, I've commented and restructured evaluate_orphanhood.py, which is the counterpart to evaluate.py from the KidSat project. This file only needs a couple of small changes to be used for predicting child deprivation for the KidSat project.
3. Added a more in-depth set of instructions for getting all the data, setting up the google cloud compute engine, training the model and getting predictions and orphanhood maps.

Here is an overall description of how we plan to predict orphanhood:
1. Get DHS data, use this to create our child deprivation indicators (we call poverty variables)
2. Aggregate DHS data, poverty variables to the cluster level and combine with GPS data
3. Associate a satellite image with each cluster
4. Finetune our DinoV2 model on the satellite imagery to predict [proportion of people who have lost a mother, ... lost a father], or this vector + the 99 dimension child deprivation vector from the KidSat project.
5. Then we add a ridge regression layer to our DinoV2 model, and fit this regression layer with the satellite imagery but now trying to predict orphanhood.
6. Now we can freely evaluate our model on a grid of satellite imagery covering a whole country, say Zambia and display a chorolopleth map of orphanhood.

## Instructions

### Intial Setup

The data step is suitably quick to run on your local computer, or this can all be done on a VM on GCP. Create a virtual environment and install all the modules in ```requirements.txt```.

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

We now need to download the satellite imagery at each of the clusters in the DHS data. For this project we have typically used 10km x 10km images, this is partially due to the jitter of the DHS data. If you are lucky, someone will have done this for you, i.e safely stored on the MLGH google drive. Otherwise you will need to extract the coordinates for each of the clusters using ```geopandas``` on the geographic Shape files. These coordinates will need to be stored in a ```DataFrame``` with columns ```name, lat, lon``` where ```name``` is the cluster ID. 

To download these satellite images you will need to code a very short script utilising ```imagery_scraping/download_imagery.py```. Firstly, update the GEE project name in the config file ```imagery_scraping/config/google_config.json```. Then you only need to load the ```DataFrame``` mentioned above for each survey, and call the ```download_imagery()``` function from ```download_imagery.py```. GEE caps the number of requests to 3000 at a time, so you will need to run the script repeatedly. It is recommended to store these satellite images at ```KidSatExt/imagery```, but this is not required.

### Google Cloud

To train our Dino model, it is necessary to utilise Google Cloud's Compute Engine. To setup this VM, follow these steps:
1. Create a project in GCP. If you are not the owner, grant yourself the appropriate IAM permissions.
1. Go to Compute Engine, select Create VM Instance.
2. Select any region, I have personally found Asia-SouthEast to have the most available GPUs.
3. On machine configuration, select GPU, then A100 40GB.
4. Increase the size of the boot disk and change the OS to Deep Learning VM with CUDA 11.8 M124 (Debian 11, Python 3.10). The important thing is making sure CUDA, torch and torchvision's versions are compatible.
5. Either allow full access to all Cloud API's or manually allow them after this setup.
6. Under Cloud Storage, create a Cloud Bucket, this is where we will upload our imagery and training/test data.
7. Upload to this by manually downloading to your local computer, then selecting Upload on the Cloud Bucket.
8. Alternatively, if the data is stored in a google drive you can using ```gcloud storage``` or ```gsutil``` to copy the files to the Bucket.

### Dino Model Training

### Next Steps

First using just orphanhood as our training labels. Secondly using orphanhood + the 99 dimension child deprivation vector.

Instructions for use:
- install modules in requirements.txt
- run main.py using the command 'python survey_processing/main.py path_to_dhs_survey_data'
  (i need to change main.py so it just outputs Zambia data)
- get the Zambia satellite images at each cluster
- run finetune_spatial_orphanhood.py to train the Dino model
- run evaluate_orphanhood.py to train the ridge regression layer of the Dino model and output predictions for orphanhood
