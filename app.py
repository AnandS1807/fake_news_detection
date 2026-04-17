from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
from collections import Counter
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

NEWS_SUBREDDITS = [
    "news",
    "worldnews",
    "inthenews",
    "politics",
    "technology",
    "business",
    "india",
    "entertainment",
    "bollywood",
]

# Initialize APIs
def init_twitter():
    bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
    consumer_key = os.getenv('TWITTER_API_KEY')
    consumer_secret = os.getenv('TWITTER_API_SECRET')
    access_token = os.getenv('TWITTER_ACCESS_TOKEN')
    access_token_secret = os.getenv('TWITTER_ACCESS_SECRET')

    if bearer_token:
        # Preferred for v2 recent search when available.
        client = tweepy.Client(
            bearer_token=bearer_token,
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
            wait_on_rate_limit=True,
        )
        return client, False

    if all([consumer_key, consumer_secret, access_token, access_token_secret]):
        # OAuth1 user context fallback for accounts without bearer token configured.
        client = tweepy.Client(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
            wait_on_rate_limit=True,
        )
        return client, True

    raise ValueError(
        "Missing Twitter credentials. Provide TWITTER_BEARER_TOKEN or full OAuth1 keys in .env"
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
    prediction_prob = float(model.predict(processed_text)[0][0])
    
    # Training labels are 0=fake, 1=real, so sigmoid > 0.5 means real.
    is_real = prediction_prob > 0.5
    confidence = prediction_prob if is_real else 1 - prediction_prob
    real_probability = prediction_prob
    fake_probability = 1 - prediction_prob
    
    result = "Real News" if is_real else "Fake News"
    confidence_percent = float(confidence) * 100
    real_probability_percent = float(real_probability) * 100
    fake_probability_percent = float(fake_probability) * 100
    
    return result, confidence_percent, real_probability_percent, fake_probability_percent
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

def build_search_terms(text, num_terms=6):
    cleaned_text = preprocess_text(text)
    tokens = [token for token in cleaned_text.split() if len(token) > 3]

    if not tokens:
        tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())

    if not tokens:
        return []

    token_counts = Counter(tokens)
    return [word for word, _ in token_counts.most_common(num_terms)]


def calculate_relevance_score(source_text, query_terms):
    if not source_text or not query_terms:
        return 0.0

    source_tokens = set(re.findall(r"[a-zA-Z]{3,}", source_text.lower()))
    if not source_tokens:
        return 0.0

    query_set = set(query_terms)
    overlap = len(source_tokens.intersection(query_set))
    return overlap / max(len(query_set), 1)


# Search Twitter for related content
def search_twitter(text, twitter_client):
    try:
        client, user_auth = twitter_client
        query_terms = build_search_terms(text)
        if not query_terms:
            return []

        query = f"({' OR '.join(query_terms[:5])}) -is:retweet lang:en"
        
        # Search Twitter
        response = client.search_recent_tweets(
            query=query,
            max_results=10,
            tweet_fields=["created_at", "public_metrics", "lang"],
            user_auth=user_auth,
        )
        
        # Extract relevant information
        if response.data:
            ranked = []
            for tweet in response.data:
                relevance = calculate_relevance_score(tweet.text, query_terms)
                metrics = getattr(tweet, "public_metrics", {}) or {}
                ranked.append({
                    "text": tweet.text,
                    "id": tweet.id,
                    "relevance": relevance,
                    "likes": metrics.get("like_count", 0),
                    "retweets": metrics.get("retweet_count", 0),
                })

            ranked.sort(
                key=lambda item: (item["relevance"], item["likes"], item["retweets"]),
                reverse=True,
            )
            return ranked[:5]
        return []
    except tweepy.errors.Forbidden as e:
        message = (
            "Twitter access forbidden. Check app/project permissions and whether recent search is enabled for your account tier."
        )
        print(f"Twitter API forbidden: {str(e)}")
        return [{"text": message, "id": ""}]
    except tweepy.errors.Unauthorized as e:
        message = "Twitter authentication failed. Recheck .env keys and regenerate tokens if needed."
        print(f"Twitter API unauthorized: {str(e)}")
        return [{"text": message, "id": ""}]
    except tweepy.errors.TooManyRequests as e:
        message = "Twitter rate limit reached. Try again in a few minutes."
        print(f"Twitter API rate limit: {str(e)}")
        return [{"text": message, "id": ""}]
    except Exception as e:
        print(f"Twitter API error: {str(e)}")
        return [{"text": f"Unable to retrieve Twitter data: {str(e)}", "id": ""}]

# Search Reddit for related content
def search_reddit(text, reddit_client):
    try:
        query_terms = build_search_terms(text)
        if not query_terms:
            return []

        search_query = " ".join(query_terms[:5])
        phrase_query = f'"{" ".join(query_terms[:3])}"' if len(query_terms) >= 2 else search_query
        seen_ids = set()
        candidates = []

        search_batches = [
            {
                "subreddits": NEWS_SUBREDDITS,
                "query": phrase_query,
                "time_filter": "year",
                "limit": 10,
                "min_relevance": 0.12,
                "scope": "focused",
            },
            {
                "subreddits": ["all"],
                "query": search_query,
                "time_filter": "year",
                "limit": 20,
                "min_relevance": 0.06,
                "scope": "broad",
            },
        ]

        for batch in search_batches:
            for subreddit_name in batch["subreddits"]:
                subreddit = reddit_client.subreddit(subreddit_name)
                submissions = subreddit.search(
                    batch["query"],
                    sort="relevance",
                    time_filter=batch["time_filter"],
                    limit=batch["limit"],
                )

                for post in submissions:
                    if post.id in seen_ids:
                        continue
                    seen_ids.add(post.id)

                    source_text = f"{post.title} {getattr(post, 'selftext', '')}"
                    relevance = calculate_relevance_score(source_text, query_terms)
                    if relevance < batch["min_relevance"]:
                        continue

                    candidates.append({
                        "title": post.title,
                        "url": post.url,
                        "score": post.score,
                        "num_comments": getattr(post, "num_comments", 0),
                        "subreddit": str(post.subreddit),
                        "relevance": round(relevance * 100, 1),
                        "scope": batch["scope"],
                    })

            if candidates:
                break

        candidates.sort(
            key=lambda item: (item["relevance"], item["num_comments"], item["score"]),
            reverse=True,
        )
        return candidates[:6]
    except Exception as e:
        print(f"Reddit API error: {str(e)}")
        return []
    
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form['text']
        text = user_input
        if user_input.startswith('http'):
            text = scrape_article(user_input)
       
        # Preprocessing and prediction logic
        processed_text = prepare_for_model(text)
        prediction, confidence, real_probability, fake_probability = make_prediction(processed_text)
       
        # Search social media
        twitter = init_twitter()
        reddit = init_reddit()
        twitter_results = search_twitter(text, twitter)
        reddit_results = search_reddit(text, reddit)
       
        return render_template(
            'index.html',
            result=prediction,
            confidence=confidence,
            real_probability=real_probability,
            fake_probability=fake_probability,
            original_input=user_input,
            twitter_posts=twitter_results,
            reddit_posts=reddit_results,
            query_terms=build_search_terms(text),
            now=datetime.now
        )
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
