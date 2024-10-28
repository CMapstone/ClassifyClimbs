from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def count_reccs(input_text, tokenizer, model):
    
    number_recommendations = 0
    for i in range (0,len(input_text)):
        inputs = tokenizer(input_text[i], return_tensors='pt')
        outputs = model(**inputs)
        predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()
        if predicted_labels[0] == 1:
            number_recommendations += 1

    return number_recommendations


def run_model_main(config):
    """
    Predict how safe a climb is from ukc comments.
    """
    #import model and tokenizer
    model_path = config['run_model']['model_input_folder']
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    #import all comments for the climb
    climb_comments=config['run_model']['climb_comments']
            
    num_recs = count_reccs(climb_comments, tokenizer, model)
    print('Number of reccomendations for this climb:')
    print(num_recs)