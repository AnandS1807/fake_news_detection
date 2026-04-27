# src/predict.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import argparse

from data_preprocessing import preprocess_text

def predict_news(text, model_path='models/fake_news_lstm.h5', tokenizer_path='models/tokenizer.pkl'):
    # Load the model and tokenizer
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Preprocess the input text
    text = preprocess_text(text)
    
    # Convert text to sequence and pad it
    max_len = 100
    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, maxlen=max_len, padding='post')
    
    # Make prediction
    prediction = (model.predict(text_pad) > 0.5).astype(int)
    return "Real News" if prediction[0] == 1 else "Fake News"


def parse_args():
    parser = argparse.ArgumentParser(description='Predict fake or real news text using trained artifacts.')
    parser.add_argument('--model', default='models/fake_news_lstm.h5', help='Path to model file.')
    parser.add_argument('--tokenizer', default='models/tokenizer.pkl', help='Path to tokenizer pickle file.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Test the model
    text = input("Enter a news article: ")
    result = predict_news(text, model_path=args.model, tokenizer_path=args.tokenizer)
    print("Prediction:", result)