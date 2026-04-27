# Fake News Detection

A Flask-based fake news classifier that uses a trained LSTM model to classify input news text (or a URL's article content) as **Fake News** or **Real News**. The app also fetches related Twitter and Reddit posts to provide extra context for the user.

This project now supports both:

- **Centralized training** (`src/train_model.py`)
- **Federated learning simulation with FedAvg** (`src/federated_train.py`)

## Project Overview

This project contains:

- A training pipeline for an LSTM text classification model.
- A federated training simulation pipeline (multi-client local training + global aggregation).
- A preprocessing module for cleaning and tokenizing text.
- A Flask web app for inference and UI rendering.
- Optional external-source context via Twitter and Reddit APIs.

## Repository Structure

```text
fake_news_detection/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── Fake.csv
│   └── True.csv
├── models/
│   ├── fake_news_lstm.h5
│   ├── fake_news_federated_lstm.h5
│   ├── federated_metrics.json
│   └── tokenizer.pkl
│   └── tokenizer_federated.pkl
├── src/
│   ├── data_preprocessing.py
│   ├── federated_train.py
│   ├── train_model.py
│   └── predict.py
└── templates/
    └── index.html
```

## Architecture

The application is organized in four layers:

1. Data layer

- `data/Fake.csv` and `data/True.csv` hold labeled news records.
- Columns observed in both files: `title`, `text`, `subject`, `date`.

2. Preprocessing layer (`src/data_preprocessing.py`)

- Loads CSV data and assigns labels:
  - Fake news -> `label = 0`
  - Real news -> `label = 1`
- Text cleaning:
  - lowercase conversion
  - non-letter character removal via regex
  - tokenization (NLTK)
  - English stopword removal
- Splits data into train/test sets.

3. Model layer (`src/train_model.py`)

- Tokenizer with vocabulary size cap (`num_words=5000`).
- Sequence conversion and padding (`max_len=100`).
- LSTM model:
  - Embedding(5000, 128)
  - LSTM(128, dropout=0.2, recurrent_dropout=0.2)
  - Dense(64, relu) + Dropout(0.5)
  - Dense(1, sigmoid)
- Compiled with Adam + binary cross-entropy.
- Trained for 5 epochs, batch size 64.
- Saves:
  - `models/fake_news_lstm.h5`
  - `models/tokenizer.pkl`

4. Application layer (`app.py` + `templates/index.html`)

- Accepts user input from a web form.
- If input starts with `http`, downloads/parses article text using `newspaper3k` (`Article`).
- Runs preprocessing + tokenization + padding.
- Gets model probability and maps to label.
- Retrieves related social posts via:
  - Twitter recent search (Tweepy)
  - Reddit search (PRAW)
- Renders prediction, confidence, timestamp, and social context in UI.

5. Federated learning layer (`src/federated_train.py`)

- Splits training data across multiple simulated clients.
- Each client trains the same LSTM locally for a small number of local epochs.
- Server aggregates client weights using **FedAvg** (weighted by client sample count).
- Server evaluates the updated global model each federated round.
- Saves federated artifacts:
  - `models/fake_news_federated_lstm.h5`
  - `models/tokenizer_federated.pkl`
  - `models/federated_metrics.json`

## Federated Learning Context for Presentation

You can present this project as a privacy-preserving fake news detector where data remains local and only model updates are shared.

### Centralized vs Federated framing

- **Centralized baseline**: one server gets all data and trains one global model.
- **Federated setup**: each organization/device trains locally, then sends model parameters (not raw text) to a coordinator.
- **Aggregation**: coordinator applies FedAvg to produce the next global model.

### Federated concepts already implemented in this repo

1. **Client partitioning**
  - `src/federated_train.py` partitions the train split into multiple clients.
2. **Local training**
  - Every client starts from global weights and runs local epochs.
3. **Server aggregation (FedAvg)**
  - Client models are averaged with sample-size weighting.
4. **Round-based optimization**
  - The process repeats for multiple communication rounds.
5. **Global evaluation tracking**
  - `models/federated_metrics.json` stores per-round loss/accuracy.

### Suggested narrative for your subject presentation

1. Problem statement:
  - Fake news data is sensitive and often decentralized across media houses/platforms.
2. Motivation:
  - Traditional centralized ML requires pooling raw data and introduces privacy/governance concerns.
3. Federated design in this project:
  - Local model updates from multiple clients + FedAvg at server.
4. Experimental setup:
  - Compare centralized model (`train_model.py`) vs federated simulation (`federated_train.py`).
5. Outcome and trade-offs:
  - Privacy and decentralization benefits, with possible convergence speed and heterogeneity challenges.

### Next federated upgrades you can claim as future work

- Non-IID client splits (simulate domain-specific clients like politics, business, regional).
- Secure aggregation / differential privacy on client updates.
- Personalized federated learning (client-specific fine-tuning).
- Communication compression (quantization/sparsification of updates).
- Real distributed runtime using frameworks like Flower or TensorFlow Federated.

## Data Flow (End to End)

### Training flow

1. `load_data()` reads fake and real CSV files.
2. Labels are attached (`0` fake, `1` real).
3. Text is normalized in `preprocess_text()`.
4. Data is split with `train_test_split(test_size=0.2, random_state=42)`.
5. Tokenizer is fit on training text only.
6. Text is converted to integer sequences.
7. Sequences are padded to length 100.
8. LSTM model is trained and evaluated.
9. Model and tokenizer artifacts are saved to `models/`.

### Inference flow (web app)

1. User submits either:

- raw news text/headline, or
- a URL to a news article.

2. If URL, app extracts article body text with `newspaper`.
3. Text goes through the same preprocessing function.
4. Tokenizer converts cleaned text to sequence.
5. Sequence is padded to fixed length 100.
6. Model outputs sigmoid probability.
7. Probability is thresholded and mapped to label + confidence.
8. App optionally fetches related Twitter/Reddit results.
9. Final result is rendered on the page.

## How the System "Scrapes" Data

There are two kinds of external data collection in current code:

1. URL article extraction

- Implemented in `scrape_article()` in `app.py`.
- Uses `newspaper.Article(url)` to download and parse article text.
- This is content extraction for a single user-provided URL, not broad web crawling.

2. Social media lookup

- Twitter: `search_recent_tweets` based on derived keywords.
- Reddit: subreddit-wide search with entity/noun-based keyword query.
- These are API queries for related context, not model training data ingestion.

Important: The model itself is trained only on local CSV datasets in `data/`.

## How Prediction Works for a Query Like "news headline"

Given an input headline (for example: `news headline`):

1. The text is lowercased and cleaned.
2. Stopwords are removed (NLTK).
3. The remaining tokens are mapped using the saved tokenizer.
4. The sequence is padded to length 100.
5. The LSTM outputs probability `p` from sigmoid.
6. Class is assigned by threshold `0.5`.

### Label-threshold behavior in code

- Training labels in `src/data_preprocessing.py`:
  - `0 = Fake`
  - `1 = Real`
- `src/predict.py` maps `prediction > 0.5` to **Real News**.
- `app.py` maps `prediction_prob > 0.5` to **Real News**.

Prediction mapping is now consistent across CLI and web app.

## Installation and Setup

## 1) Clone and enter project

```bash
git clone https://github.com/AnandS1807/fake_news_detection.git
cd fake_news_detection
```

## 2) Create virtual environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3) Install dependencies

```bash
pip install -r requirements.txt
```

If `newspaper` installation causes parser issues on your platform, you may also need:

```bash
pip install lxml lxml_html_clean
```

## 4) Download NLTK resources

The preprocessing module triggers downloads for:

- `stopwords`
- `punkt`

This happens when preprocessing module runs for the first time.

## Environment Variables (.env)

Create a `.env` file in the project root for API-based social lookup:

```env
TWITTER_API_KEY=...
TWITTER_API_SECRET=...
TWITTER_ACCESS_TOKEN=...
TWITTER_ACCESS_SECRET=...

REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=your-app-name
```

Without these values, Twitter/Reddit enrichment will fail or return fallback output.

## Run the Application

```bash
python app.py
```

Then open:

- `http://127.0.0.1:5000/`

## Retrain the Model

To retrain and overwrite model artifacts:

```bash
python src/train_model.py
```

This will:

- preprocess dataset,
- train for 5 epochs,
- evaluate on test split,
- save new `models/fake_news_lstm.h5` and `models/tokenizer.pkl`.

## Run Federated Training Simulation

```bash
python src/federated_train.py --clients 5 --rounds 3 --local-epochs 1
```

This will:

- split data into client partitions,
- train client-local LSTM models,
- aggregate via FedAvg each round,
- save `models/fake_news_federated_lstm.h5`,
- save `models/tokenizer_federated.pkl`,
- save round metrics in `models/federated_metrics.json`.

## Optional CLI Prediction

```bash
python src/predict.py
```

Enter a news text when prompted.

To test federated artifacts for inference:

```bash
python src/predict.py --model models/fake_news_federated_lstm.h5 --tokenizer models/tokenizer_federated.pkl
```

## Notes and Limitations

- The model is binary and trained on historical CSV data quality.
- External social posts are contextual signals only; they are not fused into the model score.
- URL parsing quality depends on the source webpage structure.
- There is currently no automated test suite in the repository.
- Social verification quality depends on API limits and current public discussions.

## Future Improvements

- Fix label mapping consistency in web app.
- Add unit/integration tests for preprocessing and inference.
- Version data/model artifacts with metadata.
- Add calibration metrics and confusion matrix reporting.
- Improve requirements reproducibility with pinned versions.
