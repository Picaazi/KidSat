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

### Satellite Imagery

### Google Cloud

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
