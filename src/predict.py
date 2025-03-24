# src/predict.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

from data_preprocessing import preprocess_text

def predict_news(text):
    # Load the model and tokenizer
    model = tf.keras.models.load_model('models/fake_news_lstm.h5')
    with open('models/tokenizer.pkl', 'rb') as f:
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

if __name__ == "__main__":
    # Test the model
    text = input("Enter a news article: ")
    result = predict_news(text)
    print("Prediction:", result)