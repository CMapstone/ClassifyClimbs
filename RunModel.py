from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def count_reccs(input_text, tokenizer, model):
    """
    Count how many comments in input_text report the climb to be safe.
    """     
    number_recommendations = 0
    # Iterate over all comments in input_text
    for i in range (0,len(input_text)):
        # Convert raw comment to encoding.
        inputs = tokenizer(input_text[i], return_tensors='pt')
        # classify comment as reporting climb to be 'bold' or 'safe'.
        outputs = model(**inputs)
        predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()
        # increase number_reccommendations if output class is 1 (class 1 is 'safe')
        if predicted_labels[0] == 1:
            number_recommendations += 1

    return number_recommendations


def run_model_main(config):
    """
    Predict how safe a climb is from ukc comments using a fine tuned model.
    """
    #import model and tokenizer
    model_path = config['run_model']['model_input_folder']
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    #import all comments for the climb
    climb_comments=config['run_model']['climb_comments']

    # Count the number of comments reporting climb to be safe and print out answer.        
    num_recs = count_reccs(climb_comments, tokenizer, model)
    print('Number of reccomendations for this climb:')
    print(num_recs)