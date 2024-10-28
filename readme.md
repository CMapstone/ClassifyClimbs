# Climb Classifier

This repository contains code to run the model in [RunModel.py](RunModel.py) and code to fine tune a model in [FineTuneModel.py](FineTuneModel.py).

## Fine tuning model

FineTuneModel.py trains a LLM to classify a climb as 'safe' or 'bold' based on ukc comments. The input data for this is in the Data folder, if you want to use your own data, change input_folder in confi.yaml to where your data is saved. Set model_output_folder in config.yaml to where you want the model to be saved.

## Run model

RunModel.py uses a fine-tuned model to count how many ukc users reported a climb to be safe. To run the model, copy the comments about the climb into climb_comments in config.yaml
