from pathlib import Path
from transformers import DistilBertTokenizerFast
from sklearn.model_selection import train_test_split
import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

def read_in_data(input_dir):
    """
    Read in raw text files from input folder and assign label based on subfolder. 
    """
    input_dir = Path(input_dir)
    texts = []
    labels = []
    # Iterate over the two classes.
    for label_dir in ["safe", "bold"]:
        # Iterate over each text file in each class subfolder.
        for text_file in (input_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            # Assign label 0 to everything in bold subfolder and 1 to everything in safe subfolder.
            labels.append(0 if label_dir == "bold" else 1)

    return texts, labels
    
    
def prepare_data(input_folder, tokenizer):
    """
    Transform raw data into encodings and then into datasets ready to be used as model inputs.
    """
    # Read in the raw data
    train_texts, train_labels = read_in_data(input_folder)
    # Split the dataset into a training set and validation set.
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
    # Use the tokenizer to transform raw text data into encodings.
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    # Transform encodins into datasets usable by the model to be trained.
    train_dataset = make_dataset(train_encodings, train_labels)
    val_dataset = make_dataset(val_encodings, val_labels)
    
    return train_dataset, val_dataset

class make_dataset(torch.utils.data.Dataset):
    """
    Class for a dataset in a format that is usuable by the DistilBert model.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    

def fine_tune_model_main(config):
    """
    Use a set of annotated comments from ukc to train a model to classifier climbs as 'safe' or 
    'bold'.
    """
    # Set input and output folders from the config file.
    input_folder = config['fine_tune_model']['input_folder']
    model_output_folder = config['fine_tune_model']['model_output_folder']
 
    # Set the initial model and tokenizer using the pretrained DistilBert model.
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    
    # Set the hyperparmeters used for trainin the model.
    training_args = TrainingArguments(
        output_dir = config['fine_tune_model']['results_folder'], # output directory.
        num_train_epochs = 150,              # total number of training epochs.
        per_device_train_batch_size = 16,  # batch size per device during training.
        per_device_eval_batch_size = 64,   # batch size for evaluation.
        warmup_steps = 500,                # number of warmup steps for learning rate scheduler.
        weight_decay = 0.01,               # strength of weight decay.
    )

    train_dataset, val_dataset = prepare_data(input_folder, tokenizer)
    
    # Create the trainer and train the model.
    trainer = Trainer(
        model = model,                         # the instantiated 🤗 Transformers model to be trained.
        args = training_args,                  # training arguments, defined above.
        train_dataset = train_dataset,         # training dataset.
        eval_dataset = val_dataset             # evaluation dataset.
    )
    trainer.train()

    # Save the model and tokenizer.
    model.save_pretrained(model_output_folder)
    tokenizer.save_pretrained(model_output_folder)

    return
