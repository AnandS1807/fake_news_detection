import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from data_preprocessing import load_data, preprocess_text


def build_lstm_model(max_len=100, vocab_size=5000):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_tokenizer(texts, vocab_size=5000):
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def encode_texts(tokenizer, texts, max_len=100):
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=max_len, padding="post")


def preprocess_dataset():
    data = load_data().copy()
    data["text"] = data["text"].astype(str).apply(preprocess_text)
    return data


def split_clients(X, y, num_clients=5, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)

    shuffled_X = X[indices]
    shuffled_y = y[indices]

    x_parts = np.array_split(shuffled_X, num_clients)
    y_parts = np.array_split(shuffled_y, num_clients)
    return list(zip(x_parts, y_parts))


def weighted_average(weights_list, sample_counts):
    total_samples = float(np.sum(sample_counts))
    averaged = []
    for layers in zip(*weights_list):
        layer_sum = np.zeros_like(layers[0])
        for layer_weights, n in zip(layers, sample_counts):
            layer_sum += layer_weights * (n / total_samples)
        averaged.append(layer_sum)
    return averaged


def federated_train(
    num_clients=5,
    rounds=3,
    local_epochs=1,
    batch_size=64,
    max_len=100,
    vocab_size=5000,
    test_size=0.2,
    seed=42,
):
    print("Preparing dataset...")
    data = preprocess_dataset()
    X = data["text"].values
    y = data["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    tokenizer = build_tokenizer(X_train, vocab_size=vocab_size)
    X_train_pad = encode_texts(tokenizer, X_train, max_len=max_len)
    X_test_pad = encode_texts(tokenizer, X_test, max_len=max_len)

    clients = split_clients(X_train_pad, y_train, num_clients=num_clients, seed=seed)

    global_model = build_lstm_model(max_len=max_len, vocab_size=vocab_size)

    history = []

    for round_idx in range(1, rounds + 1):
        print(f"\n--- Federated Round {round_idx}/{rounds} ---")
        global_weights = global_model.get_weights()

        client_weights = []
        sample_counts = []

        for client_idx, (x_client, y_client) in enumerate(clients, start=1):
            local_model = build_lstm_model(max_len=max_len, vocab_size=vocab_size)
            local_model.set_weights(global_weights)

            local_model.fit(
                x_client,
                y_client,
                epochs=local_epochs,
                batch_size=batch_size,
                verbose=0,
            )

            client_weights.append(local_model.get_weights())
            sample_counts.append(len(x_client))
            print(f"Client {client_idx}: trained on {len(x_client)} samples")

        new_global_weights = weighted_average(client_weights, sample_counts)
        global_model.set_weights(new_global_weights)

        test_loss, test_acc = global_model.evaluate(X_test_pad, y_test, verbose=0)
        history.append({"round": round_idx, "test_loss": float(test_loss), "test_accuracy": float(test_acc)})
        print(f"Round {round_idx} test accuracy: {test_acc:.4f}")

    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "fake_news_federated_lstm.h5"
    tokenizer_path = output_dir / "tokenizer_federated.pkl"
    metrics_path = output_dir / "federated_metrics.json"

    global_model.save(model_path)
    with tokenizer_path.open("wb") as f:
        pickle.dump(tokenizer, f)
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("\nFederated model artifacts saved:")
    print(f"- {model_path}")
    print(f"- {tokenizer_path}")
    print(f"- {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Federated LSTM training simulation for fake news detection.")
    parser.add_argument("--clients", type=int, default=5, help="Number of federated clients.")
    parser.add_argument("--rounds", type=int, default=3, help="Number of federated rounds.")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per client per round.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for local client training.")
    parser.add_argument("--max-len", type=int, default=100, help="Maximum token sequence length.")
    parser.add_argument("--vocab-size", type=int, default=5000, help="Tokenizer vocabulary size.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    federated_train(
        num_clients=args.clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        vocab_size=args.vocab_size,
    )
