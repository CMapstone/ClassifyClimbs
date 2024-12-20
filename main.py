from nbformat import read
import yaml

from RunModel import run_model_main
from FineTuneModel import fine_tune_model_main


def read_config():

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    return config


def run_main(run_model, fine_tune_model, config):
    
    if fine_tune_model:
        fine_tune_model_main(config)

    if run_model:
        run_model_main(config)
     
    return


if __name__=='__main__':

    config = read_config()
    # Set the tasks you want to run to 'True' and leave the others as 'False'.
    fine_tune_model = True
    run_model = True

    run_main( run_model, fine_tune_model, config)

