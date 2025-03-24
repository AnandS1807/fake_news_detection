# src/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report

from data_preprocessing import load_data, preprocess_data

def train_deep_learning_model(X_train, X_test, y_train, y_test):
    # Tokenize the text data
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    
    # Convert text to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences to ensure uniform input size
    max_len = 100  # Maximum sequence length
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')
    
    # Build the LSTM model
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_len))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_split=0.2)
    
    # Evaluate the model
    y_pred = (model.predict(X_test_pad) > 0.5).astype(int)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Save the model and tokenizer
    model.save('models/fake_news_lstm.h5')
    import pickle
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print("Model and tokenizer saved!")

if __name__ == "__main__":
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)
    train_deep_learning_model(X_train, X_test, y_train, y_test)