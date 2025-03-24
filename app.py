# src/app.py (Updated for LSTM Model)
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load the LSTM model and tokenizer
model = tf.keras.models.load_model('models/fake_news_lstm.h5')
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    import re
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        text = preprocess_text(text)
        
        # Convert text to sequence and pad it
        max_len = 100
        text_seq = tokenizer.texts_to_sequences([text])
        text_pad = pad_sequences(text_seq, maxlen=max_len, padding='post')
        
        # Make prediction
        prediction = (model.predict(text_pad) > 0.5).astype(int)
        result = "Real News" if prediction[0] == 1 else "Fake News"
        return render_template('index.html', result=result)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)