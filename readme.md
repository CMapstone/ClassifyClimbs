# Climb Classifier

This repository contains a pipeline to automatically assess how well protected a climb is using UKC logbook notes. First, a pre-trained large language model, DistilBERT, is trained to classify a climb as either 'safe' or 'bold' based on a single logbook comment using the code in [FineTuneModel.py](FineTuneModel.py). Then, the code in [RunModel.py](RunModel.py) can be used to run this model on every comment for a climb and count the number of comments that suggest the climb is safe. 

## Fine tuning model

FineTuneModel.py trains a LLM to classify a climb as 'safe' or 'bold' based on ukc logbook notes. There is example input data for this in the Data folder. Before running this code, set model_output_folder in config.yaml to where you want the model to be saved and input_folder to the location of your data for training. 

## Running model

RunModel.py uses a fine-tuned model to count how many ukc users reported a climb to be safe. To run the model, copy the comments about the climb into climb_comments in config.yaml and set model_input_folder to the location of your trained model.
