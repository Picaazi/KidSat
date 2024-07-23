# KidSatExt
We aim to use the DinoV2 model to predict orphanhood in Zambia in 2018, 2020 and 2023. With the aim to compare this to data we have in the Sinazongwe District.
First using just orphanhood as our training labels. Secondly using orphanhood + the 99 dimension child deprivation vector.

Instructions for use:
- install modules in requirements.txt
- run main.py using the command 'python survey_processing/main.py path_to_dhs_survey_data'
  (i need to change main.py so it just outputs Zambia data)
- get the Zambia satellite images at each cluster
- run finetune_spatial_orphanhood.py to train the Dino model
- run evaluate_orphanhood.py to train the ridge regression layer of the Dino model and output predictions for orphanhood
