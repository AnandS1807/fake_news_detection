from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from newspaper import Article
import tweepy
import praw
import os
from dotenv import load_dotenv
from datetime import datetime
from src.data_preprocessing import preprocess_text

# Load environment variables
load_dotenv()
app = Flask(__name__)

# Load model and tokenizer
model = tf.keras.models.load_model('models/fake_news_lstm.h5')
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Initialize APIs
def init_twitter():
    return tweepy.Client(
        consumer_key=os.getenv('TWITTER_API_KEY'),
        consumer_secret=os.getenv('TWITTER_API_SECRET'),
        access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
        access_token_secret=os.getenv('TWITTER_ACCESS_SECRET')
    )

def init_reddit():
    return praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent=os.getenv('REDDIT_USER_AGENT')
    )
# Function to scrape article content from URL
def scrape_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"Error scraping article: {str(e)}"

# Function to prepare text for model prediction
def prepare_for_model(text):
    # First use your existing preprocessing function
    cleaned_text = preprocess_text(text)
    
    # Then tokenize and pad for the model
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    max_length = 100  # Adjust based on your model's expected input
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    
    return padded_sequences

# Make prediction using the loaded model
def make_prediction(processed_text):
    # Get raw prediction
    prediction_prob = model.predict(processed_text)[0][0]
    
    # Convert to binary classification with confidence
    is_fake = prediction_prob > 0.5
    confidence = prediction_prob if is_fake else 1 - prediction_prob
    
    result = "Fake News" if is_fake else "Real News"
    confidence_percent = float(confidence) * 100
    
    return result, confidence_percent
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_keywords(text, num_keywords=5):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    
    # We need a collection of documents for TF-IDF to work properly
    # Here we split the text into sentences to create that collection
    sentences = text.split('.')
    
    # Remove empty sentences
    sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 10]
    
    if not sentences:
        return text.split()[:num_keywords]  # Fallback to first few words
    
    # Fit and transform the sentences
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate the average TF-IDF score for each word across all sentences
    avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
    
    # Create a dictionary of word: score
    word_scores = {feature_names[i]: avg_scores[i] for i in range(len(feature_names))}
    
    # Sort words by score and get top keywords
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, score in sorted_words[:num_keywords]]
    
    return keywords

def extract_keywords_with_entities(text, num_keywords=5):
    try:
        import spacy
        
        # Load spaCy model
        nlp = spacy.load("en_core_web_sm")
        
        # Process the text
        doc = nlp(text[:5000])  # Limit text length to avoid processing too much
        
        # Extract named entities
        entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT']]
        
        # Extract nouns and proper nouns
        nouns = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 3]
        
        # Combine entities and nouns, prioritizing entities
        keywords = []
        keywords.extend(entities)
        keywords.extend([noun for noun in nouns if noun not in keywords])
        
        # Return unique keywords
        return list(dict.fromkeys(keywords))[:num_keywords]
    
    except (ImportError, OSError):
        # Fallback if spaCy is not available or model not installed
        return text.split()[:num_keywords]

def extract_keywords_simple(text, num_keywords=5):
    import re
    from collections import Counter
    
    # Convert to lowercase and remove special characters
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    # Split into words
    words = text.split()
    
    # Remove common stop words
    stop_words = {'the', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
                  'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                  'had', 'do', 'does', 'did', 'but', 'or', 'as', 'if', 'while', 'because', 
                  'then', 'also', 'so', 'that', 'this', 'these', 'those', 'it', 'its', 'from'}
    
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Get the most common words
    common_words = [word for word, count in word_counts.most_common(num_keywords)]
    
    return common_words
# Search Twitter for related content
def search_twitter(text, twitter_client):
    try:
        # Extract keywords (simple implementation)
        keywords = ' '.join(text.split()[:10])
        
        # Search Twitter
        response = twitter_client.search_recent_tweets(query=keywords, max_results=5)
        
        # Extract relevant information
        if response.data:
            return [{"text": tweet.text, "id": tweet.id} for tweet in response.data]
        return []
    except Exception as e:
        print(f"Twitter API error: {str(e)}")
        return [{"text": "Unable to retrieve Twitter data due to API limitations.", "id": ""}]

# Search Reddit for related content
def search_reddit(text, reddit_client):
    try:
        # Extract better keywords using spaCy NER and POS tagging
        keywords = extract_keywords_with_entities(text, num_keywords=5)
        
        # Join keywords with OR for broader search
        search_query = ' OR '.join(keywords)
        
        # Search Reddit
        submissions = reddit_client.subreddit("all").search(search_query, limit=5)
        
        # Extract relevant information
        return [{"title": post.title, "url": post.url, "score": post.score} 
                for post in submissions]
    except Exception as e:
        print(f"Reddit API error: {str(e)}")
        return []
    
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        if text.startswith('http'):
            text = scrape_article(text)
       
        # Preprocessing and prediction logic
        processed_text = prepare_for_model(text)
        prediction, confidence = make_prediction(processed_text)
       
        # Search social media
        twitter = init_twitter()
        reddit = init_reddit()
        twitter_results = search_twitter(text, twitter)
        reddit_results = search_reddit(text, reddit)
       
        return render_template(
            'index.html',
            result=prediction,
            confidence=confidence,
            twitter_posts=twitter_results,
            reddit_posts=reddit_results,
            now=datetime.now
        )
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
