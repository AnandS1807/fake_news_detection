# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')

def load_data():
    # Load fake and real news datasets
    fake_news = pd.read_csv('data/Fake.csv')
    real_news = pd.read_csv('data/True.csv')
    
    # Add labels
    fake_news['label'] = 0  # 0 for fake news
    real_news['label'] = 1   # 1 for real news
    
    # Combine datasets
    data = pd.concat([fake_news, real_news], axis=0)
    return data

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def preprocess_data(data):
    # Preprocess text data
    data['text'] = data['text'].apply(preprocess_text)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)
    print("Data preprocessing complete!")