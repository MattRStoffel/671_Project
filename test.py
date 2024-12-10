# TO run this you will need to have the ollama server running on your loacal machine using ollama serve
import pickle
import pandas as pd
import torch
from ollama import chat
from ollama import ChatResponse
from data import preprocess_text
import util
import data

def get_negation(text):
    response: ChatResponse = chat(model='phi3', messages=[
        {
            'role': 'user',
            'content': 'negate the following sentence and do not say anything else it is important that you only try to make it sounnd like its coming from the opposite political party. Try to only change one thing to make that happen. THE MAIN GOAL IS TO CHANGE AS LITTLE AS POSSIBLE WHILE STILL MAKING IT THE OPOSITE PARTY: ' + text,
        },
    ])
    return response['message']['content']

with open('results.pkl', 'rb') as f:
    results = pickle.load(f)

df = pd.read_csv('ExtractedTweets.csv')

selected_epoch = 4
selected_batchsize = 500
selected_learning_rate = 0.001

model, training_loss, validation_accuracy = results[(selected_epoch, selected_batchsize, selected_learning_rate)]

import torch

def get_class(tensor_output):
    predicted_class_index = torch.argmax(tensor_output, dim=1).item()

    if predicted_class_index == 1:
        return 1
    else:
        return 0


# Function to test the model with the negated tweets
def test_model_with_negated_tweets(df, model):
    model.eval()  # Make sure the model is in evaluation mode
    
    test_results = []
    for index, tweet in df['Tweet'][:9000][20:].items():
        # Get the negated tweet from the API
        negated_tweet = get_negation(tweet)

        # Preprocess both original and negated tweets
        processed_tweet = preprocess_text(tweet)
        processed_negated = preprocess_text(negated_tweet)

        # Convert processed text into tensors
        # Assuming preprocess_text returns a list, we need to convert it into a tensor
        processed_tweet_tensor = torch.tensor(processed_tweet, dtype=torch.long).unsqueeze(0)  # Add batch dimension
        processed_negated_tensor = torch.tensor(processed_negated, dtype=torch.long).unsqueeze(0)  # Add batch dimension

        try:
            # Run the model on both the original and negated tweets
          with torch.no_grad():  # Disable gradient calculation during inference
              model_official = get_class(model(processed_tweet_tensor))  # Pass tensor to model
              model_negated = get_class(model(processed_negated_tensor))  # Pass tensor to model
        except Exception as e:
            continue 
        
        x = (df.loc[index, 'Party'], model_official, model_negated)
        test_results.append(x)
        util.save_results(test_results, "reverse.pkl")

        
        # Print results
        print(f"Original: {tweet}")
        print(f"Negated: {negated_tweet}")
        print(f"True label output:     {df.loc[index, 'Party']}")
        print(f"Model official output: {model_official}")
        print(f"Model negated output:  {model_negated}")
        print("-" * 50)

# Call the function to test the model with the negated tweets
test_model_with_negated_tweets(df, model)

