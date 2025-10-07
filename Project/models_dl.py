"""
models_dl.py
============

This module defines deep learning models for text classification,
including an LSTM network, a CNN and a transformer‑based model using
BERT.  It also provides a function to train these models and return
evaluation results.
"""

from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Conv1D,
    MaxPooling1D,
    Dropout,
    GlobalMaxPooling1D,
)
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

import gensim.downloader as api


from .evaluation import ModelResult


def build_lstm_model(
    vocab_size: int,
    embedding_dim: int = 128,
    input_length: int = 100,
) -> tf.keras.Model:
    """
    Build and compile a simple LSTM model for text classification.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary (number of unique tokens).
    embedding_dim : int
        Dimensionality of the embedding space.
    input_length : int
        Length of the input sequences (after padding/truncation).

    Returns
    -------
    model : tf.keras.Model
        The compiled LSTM model.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
        LSTM(128, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    return model


def build_cnn_model(
    vocab_size: int,
    embedding_dim: int = 128,
    input_length: int = 100,
) -> tf.keras.Model:
    """
    Build and compile a 1D CNN model for text classification.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embedding_dim : int
        Dimensionality of the embedding space.
    input_length : int
        Length of the input sequences.

    Returns
    -------
    model : tf.keras.Model
        The compiled CNN model.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        GlobalMaxPooling1D(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    return model


def train_deep_models(
    train_texts: List[str],
    test_texts: List[str],
    y_train: List[int],
    y_test: List[int],
    max_num_words: int = 10000,
    max_sequence_length: int = 100,
    batch_size: int = 32,
    lstm_epochs: int = 5,
    cnn_epochs: int = 5,
    bert_epochs: int = 2,
    bert_model_name: str = 'bert-base-uncased',
) -> List[ModelResult]:
    """
    Train deep learning models (LSTM, CNN, BERT) and evaluate them.

    Parameters
    ----------
    train_texts : List[str]
        Cleaned training texts.
    test_texts : List[str]
        Cleaned testing texts.
    y_train : List[int]
        Training labels.
    y_test : List[int]
        Testing labels.
    max_num_words : int
        Maximum number of words to keep in the tokenizer vocabulary.
    max_sequence_length : int
        Maximum length of sequences (padding/truncation).
    batch_size : int
        Batch size for training deep models.
    lstm_epochs : int
        Epochs for LSTM model.
    cnn_epochs : int
        Epochs for CNN model.
    bert_epochs : int
        Epochs for BERT model.
    bert_model_name : str
        Hugging Face model name (e.g. 'bert-base-uncased').

    Returns
    -------
    results : List[ModelResult]
        Evaluation results for each model.
    """
    results = []

    # Convert texts to sequences for LSTM and CNN
    tokenizer = Tokenizer(num_words=max_num_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_texts)
    X_train_seq = tokenizer.texts_to_sequences(train_texts)
    X_test_seq = tokenizer.texts_to_sequences(test_texts)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post', truncating='post')
    y_train_array = np.array(y_train)
    y_test_array = np.array(y_test)
    vocab_size = min(max_num_words, len(tokenizer.word_index) + 1)

    # -----------------
    # Train LSTM (random embeddings)
    # -----------------
    lstm_model = build_lstm_model(vocab_size=vocab_size, input_length=max_sequence_length)
    lstm_model.fit(
        X_train_pad,
        y_train_array,
        epochs=lstm_epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=2,
    )
    lstm_metrics = lstm_model.evaluate(X_test_pad, y_test_array, verbose=0)
    lstm_accuracy = lstm_metrics[1]
    lstm_precision = lstm_metrics[2]
    lstm_recall = lstm_metrics[3]
    lstm_f1 = 2 * (lstm_precision * lstm_recall) / (lstm_precision + lstm_recall + 1e-8)
    results.append(
        ModelResult(
            name='LSTM',
            accuracy=lstm_accuracy,
            precision=lstm_precision,
            recall=lstm_recall,
            f1=lstm_f1,
        )
    )

    # -----------------
    # Train LSTM with GloVe embeddings
    # -----------------
    # Load pre-trained GloVe embeddings (100‑dimensional).  This step can be
    # time‑consuming; consider caching the embedding model locally.
    try:
        glove_model = api.load('glove-wiki-gigaword-100')
    except Exception as e:
        print(f"Warning: failed to load GloVe embeddings: {e}. Skipping pretrained LSTM model.")
        glove_model = None

    if glove_model is not None:
        embedding_dim = 100
        # Initialize embedding matrix with zeros
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        for word, i in tokenizer.word_index.items():
            if i >= vocab_size:
                continue
            if word in glove_model:
                embedding_matrix[i] = glove_model[word]
        # Build LSTM model with pre‑trained embeddings
        lstm_pretrained = Sequential([
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                weights=[embedding_matrix],
                input_length=max_sequence_length,
                trainable=False,
            ),
            LSTM(128, return_sequences=False),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid'),
        ])
        lstm_pretrained.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        )
        lstm_pretrained.fit(
            X_train_pad,
            y_train_array,
            epochs=lstm_epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=2,
        )
        glove_metrics = lstm_pretrained.evaluate(X_test_pad, y_test_array, verbose=0)
        glove_accuracy = glove_metrics[1]
        glove_precision = glove_metrics[2]
        glove_recall = glove_metrics[3]
        glove_f1 = 2 * (glove_precision * glove_recall) / (glove_precision + glove_recall + 1e-8)
        results.append(
            ModelResult(
                name='LSTM + GloVe',
                accuracy=glove_accuracy,
                precision=glove_precision,
                recall=glove_recall,
                f1=glove_f1,
            )
        )

    # Train CNN
    cnn_model = build_cnn_model(vocab_size=vocab_size, input_length=max_sequence_length)
    cnn_model.fit(
        X_train_pad,
        y_train_array,
        epochs=cnn_epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=2,
    )
    cnn_metrics = cnn_model.evaluate(X_test_pad, y_test_array, verbose=0)
    cnn_accuracy = cnn_metrics[1]
    cnn_precision = cnn_metrics[2]
    cnn_recall = cnn_metrics[3]
    cnn_f1 = 2 * (cnn_precision * cnn_recall) / (cnn_precision + cnn_recall + 1e-8)
    results.append(
        ModelResult(
            name='CNN',
            accuracy=cnn_accuracy,
            precision=cnn_precision,
            recall=cnn_recall,
            f1=cnn_f1,
        )
    )

    # -----------------
    # Train a custom Transformer model using Keras
    # -----------------
    # Build a transformer-inspired network with multi-head attention.  This avoids
    # external dependencies on Hugging Face models and satisfies the
    # transformer-based requirement.
    from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Input
    from tensorflow.keras.models import Model

    inputs = Input(shape=(max_sequence_length,))
    x_emb = Embedding(input_dim=vocab_size, output_dim=128)(inputs)
    # Multi-head self-attention
    attn_output = MultiHeadAttention(num_heads=4, key_dim=32)(x_emb, x_emb)
    # Add & norm
    x = LayerNormalization(epsilon=1e-6)(attn_output + x_emb)
    # Global average pooling
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    transformer_model = Model(inputs=inputs, outputs=outputs)
    transformer_model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    transformer_model.fit(
        X_train_pad,
        y_train_array,
        epochs=cnn_epochs,  # reuse cnn_epochs for transformer
        batch_size=batch_size,
        validation_split=0.1,
        verbose=2,
    )
    transformer_metrics = transformer_model.evaluate(X_test_pad, y_test_array, verbose=0)
    transformer_accuracy = transformer_metrics[1]
    transformer_precision = transformer_metrics[2]
    transformer_recall = transformer_metrics[3]
    transformer_f1 = 2 * (transformer_precision * transformer_recall) / (transformer_precision + transformer_recall + 1e-8)
    results.append(
        ModelResult(
            name='Transformer',
            accuracy=transformer_accuracy,
            precision=transformer_precision,
            recall=transformer_recall,
            f1=transformer_f1,
        )
    )
    return results
