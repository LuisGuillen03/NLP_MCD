"""
Final Project for NLP Text Classification
========================================

This script implements a full pipeline for a text classification project.  It
downloads a publicly available dataset, performs necessary preprocessing,
balances the dataset (if required), trains both traditional machine learning
models and deep learning models, evaluates them on a held‑out test set, and
generates comparative tables and charts.  The code is organized into
functions to make it easy to understand and extend.  It is provided as an
example for educational purposes and can be executed end‑to‑end by running
`python final_project_nlp.py` from the command line.  Note that this script
does not automatically run upon import; you must call the main function
explicitly to execute the workflow.

Requirements
------------
Before running the script, ensure that you have installed the following
packages:

* pandas
* numpy
* scikit‑learn
* nltk
* imbalanced‑learn
* matplotlib
* seaborn
* datasets
* tensorflow (>=2.0)
* transformers

You can install them via pip:

```bash
pip install pandas numpy scikit-learn nltk imbalanced-learn matplotlib seaborn datasets tensorflow transformers
```

Dataset
-------
This project uses the SMS Spam Collection dataset, a well‑known dataset for
spam detection consisting of 5,574 text messages labeled as "ham" (legitimate)
or "spam".  It is available via the `datasets` library under the name
`ucirvine/sms_spam`【192316555573128†L80-L95】.  To adhere to the project
requirement of using at most 5,000 samples, the script randomly samples
5,000 rows from the dataset after loading it.

Instructions
------------
1. Download the dataset using the `load_dataset` function from the
   `datasets` library.
2. Preprocess the text:
   * Lowercase the messages
   * Remove punctuation, numbers, and extra whitespace
   * Remove stopwords using NLTK's stopword list
   * Optionally apply lemmatization (commented out by default)
3. Balance the dataset using either under‑sampling or over‑sampling; this
   script uses `RandomUnderSampler` from `imbalanced_learn` to equalize the
   number of "ham" and "spam" messages.
4. Split the data into training and test sets.
5. Create TF‑IDF representations of the cleaned text for traditional ML models.
6. Train three traditional models: logistic regression, linear SVM, and
   random forest.
7. Evaluate these models using accuracy, precision, recall and F1‑score.
8. For deep learning models, create sequences of integer‑encoded tokens using
   Keras' `Tokenizer`, pad them, and build two networks:
   * LSTM model with an embedding layer initialized randomly.
   * 1D CNN model for text classification.
   In addition, a transformer‑based model using BERT is implemented via the
   `transformers` library.
9. Evaluate the deep learning models and collect the same set of metrics.
10. Generate comparison tables and bar charts summarizing the performance of
    all models.

Note: Training deep learning models can be computationally intensive.  For
quick experimentation, consider reducing the number of epochs or using a
smaller subset of data.
"""

import os
import re
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords

from datasets import load_dataset

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

from transformers import (
    BertTokenizerFast,
    TFAutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    create_optimizer,
)
from transformers import TrainingArguments, Trainer


# Ensure NLTK resources are downloaded
nltk.download('stopwords')


@dataclass
class ModelResult:
    """Simple container for model evaluation results."""

    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float


def download_and_sample_dataset(seed: int = 42, sample_size: int = 5000) -> pd.DataFrame:
    """
    Download the SMS Spam Collection dataset and return a pandas DataFrame
    limited to `sample_size` samples.  If the dataset contains fewer samples
    than `sample_size`, the full dataset is returned.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility when sampling.
    sample_size : int
        Maximum number of rows to return.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns 'text' and 'label'.
    """
    dataset = load_dataset('ucirvine/sms_spam', split='train')
    # Convert to pandas DataFrame
    df = pd.DataFrame(dataset)
    # The dataset uses 'sms' and 'label' columns; rename 'sms' to 'text' for convenience
    df = df.rename(columns={'sms': 'text'})
    # If the dataset is larger than requested, sample without replacement
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return df


def clean_text(text: str, stop_words: set) -> str:
    """
    Clean a single text string by lowercasing, removing non‑alphabetic characters,
    and stripping stopwords.  Additional preprocessing steps can be added here.

    Parameters
    ----------
    text : str
        The raw text message to clean.
    stop_words : set
        A set of stopwords to remove from the text.

    Returns
    -------
    cleaned : str
        The cleaned and tokenized text joined back into a single string.
    """
    # Lowercase the text
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', ' ', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenize by splitting on whitespace
    tokens = text.split()
    # Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    # Optionally, apply lemmatization (commented out to keep dependencies minimal)
    # from nltk.stem import WordNetLemmatizer
    # nltk.download('wordnet')
    # lemmatizer = WordNetLemmatizer()
    # tokens = [lemmatizer.lemmatize(t) for t in tokens]
    cleaned = ' '.join(tokens)
    return cleaned


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning to the entire DataFrame and return a new DataFrame with
    the cleaned text.  Stopwords are taken from NLTK's English corpus.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns 'text' and 'label'.

    Returns
    -------
    processed_df : pandas.DataFrame
        DataFrame with an additional 'clean_text' column containing cleaned text.
    """
    english_stopwords = set(stopwords.words('english'))
    df['clean_text'] = df['text'].apply(lambda x: clean_text(x, english_stopwords))
    return df


def balance_dataset(
    texts: List[str], labels: List[int], random_state: int = 42
) -> Tuple[List[str], List[int]]:
    """
    Balance the dataset using random under‑sampling so that each class has the
    same number of samples.  Under‑sampling is chosen because it is simple to
    implement and prevents synthetic generation of data, which may introduce
    artifacts.  For experiments that can benefit from more sophisticated
    balancing, consider using `SMOTE` or `RandomOverSampler` instead.

    Parameters
    ----------
    texts : List[str]
        List of cleaned text messages.
    labels : List[int]
        Corresponding list of integer labels.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    balanced_texts : List[str]
        Text messages after balancing.
    balanced_labels : List[int]
        Labels after balancing.
    """
    rus = RandomUnderSampler(random_state=random_state)
    # Convert lists to DataFrame for compatibility with imbalanced‑learn API
    X = pd.DataFrame({'text': texts})
    y = pd.Series(labels)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return X_resampled['text'].tolist(), y_resampled.tolist()


def vectorize_texts(train_texts: List[str], test_texts: List[str]) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    Convert lists of text messages into TF‑IDF feature matrices for training
    and testing.  The same vectorizer is fit on the training data and applied
    to the test data to avoid information leakage.

    Parameters
    ----------
    train_texts : List[str]
        Cleaned text messages for training.
    test_texts : List[str]
        Cleaned text messages for testing.

    Returns
    -------
    X_train : numpy.ndarray
        TF‑IDF feature matrix for the training texts.
    X_test : numpy.ndarray
        TF‑IDF feature matrix for the test texts.
    vectorizer : TfidfVectorizer
        The fitted vectorizer (useful for transforming new data).
    """
    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), sublinear_tf=True, stop_words='english'
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer


def train_ml_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: List[int],
    y_test: List[int],
) -> List[ModelResult]:
    """
    Train three traditional machine learning models (Logistic Regression,
    Linear SVM, Random Forest) on the provided TF‑IDF features and evaluate
    them on the test set.  Returns a list of `ModelResult` objects with
    evaluation metrics.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training feature matrix.
    X_test : numpy.ndarray
        Testing feature matrix.
    y_train : List[int]
        Training labels.
    y_test : List[int]
        Testing labels.

    Returns
    -------
    results : List[ModelResult]
        List of results for each model.
    """
    results = []
    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    results.append(
        ModelResult(
            name='Logistic Regression',
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
        )
    )

    # Linear SVM
    svm_model = LinearSVC()
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    results.append(
        ModelResult(
            name='Linear SVM',
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
        )
    )

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    results.append(
        ModelResult(
            name='Random Forest',
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
        )
    )
    return results


def build_lstm_model(
    vocab_size: int,
    embedding_dim: int = 128,
    input_length: int = 100,
) -> tf.keras.Model:
    """
    Build and compile a simple LSTM model for text classification.  The
    embedding layer is initialized randomly.  You can modify this function
    to load pre‑trained embeddings (e.g., GloVe or Word2Vec) by setting
    the `weights` argument of the Embedding layer.

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
    Build and compile a 1D CNN model for text classification.  The network
    uses convolutional filters followed by global max pooling to extract
    features from the sequence of embeddings.

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
    Train deep learning models (LSTM, CNN, and BERT) on the provided data and
    evaluate them on the test set.  The function returns a list of
    `ModelResult` objects containing evaluation metrics for each model.

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
        Maximum length of sequences (shorter sequences are padded and longer
        ones are truncated).
    batch_size : int
        Batch size for training deep models.
    lstm_epochs : int
        Number of epochs to train the LSTM model.
    cnn_epochs : int
        Number of epochs to train the CNN model.
    bert_epochs : int
        Number of epochs to fine‑tune the BERT model.
    bert_model_name : str
        Name of the pre‑trained BERT model to use (from Hugging Face).  For
        example, 'bert-base-uncased' or 'bert-base-multilingual-cased'.

    Returns
    -------
    results : List[ModelResult]
        List of results for each deep learning model.
    """
    results = []

    # Tokenization and sequence padding for LSTM and CNN
    tokenizer = Tokenizer(num_words=max_num_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_texts)
    X_train_seq = tokenizer.texts_to_sequences(train_texts)
    X_test_seq = tokenizer.texts_to_sequences(test_texts)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post', truncating='post')

    y_train_array = np.array(y_train)
    y_test_array = np.array(y_test)

    vocab_size = min(max_num_words, len(tokenizer.word_index) + 1)

    # LSTM Model
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
    # The evaluate method returns [loss, accuracy, precision, recall]
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

    # CNN Model
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

    # BERT Model using Hugging Face Transformers
    # Tokenize the text using the BERT tokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    # Tokenize and encode the texts into input IDs and attention masks
    train_encodings = bert_tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=max_sequence_length,
        return_tensors='tf',
    )
    test_encodings = bert_tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=max_sequence_length,
        return_tensors='tf',
    )
    # Build dataset objects for TensorFlow
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        y_train_array,
    )).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        y_test_array,
    )).batch(batch_size)

    # Load the pre‑trained model
    bert_model = TFAutoModelForSequenceClassification.from_pretrained(
        bert_model_name,
        num_labels=2,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    bert_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    bert_model.fit(
        train_dataset,
        epochs=bert_epochs,
        validation_data=test_dataset,
        verbose=2,
    )
    # Evaluate BERT model
    bert_metrics = bert_model.evaluate(test_dataset, verbose=0)
    # bert_metrics: [loss, accuracy, precision, recall]
    bert_accuracy = bert_metrics[1]
    bert_precision = bert_metrics[2]
    bert_recall = bert_metrics[3]
    bert_f1 = 2 * (bert_precision * bert_recall) / (bert_precision + bert_recall + 1e-8)
    results.append(
        ModelResult(
            name='BERT',
            accuracy=bert_accuracy,
            precision=bert_precision,
            recall=bert_recall,
            f1=bert_f1,
        )
    )

    return results


def plot_results(results: List[ModelResult], title: str = 'Model Performance Comparison') -> None:
    """
    Plot bar charts comparing accuracy, precision, recall and F1‑score for
    multiple models.  Saves the plot as a PNG file in the current working
    directory.

    Parameters
    ----------
    results : List[ModelResult]
        List of model evaluation results.
    title : str
        Title for the plot.
    """
    metrics_df = pd.DataFrame([
        {
            'Model': r.name,
            'Accuracy': r.accuracy,
            'Precision': r.precision,
            'Recall': r.recall,
            'F1‑Score': r.f1,
        }
        for r in results
    ])
    metrics_df_melted = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df_melted, x='Model', y='Score', hue='Metric')
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png')
    plt.close()


def main() -> None:
    """
    Execute the full workflow: download and preprocess the data, balance it,
    train models, evaluate them, and generate comparison plots.  Results are
    printed to the console and the bar chart is saved to disk.
    """
    # 1. Data selection
    df = download_and_sample_dataset(sample_size=5000)

    # 2. Preprocessing
    df = preprocess_dataframe(df)

    # 3. Balance the dataset
    balanced_texts, balanced_labels = balance_dataset(df['clean_text'].tolist(), df['label'].tolist())

    # 4. Split into train and test sets
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        balanced_texts,
        balanced_labels,
        test_size=0.2,
        random_state=42,
        stratify=balanced_labels,
    )

    # 5. Traditional ML models
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_texts(X_train_texts, X_test_texts)
    ml_results = train_ml_models(X_train_tfidf, X_test_tfidf, y_train, y_test)

    # 6. Deep learning models
    deep_results = train_deep_models(
        train_texts=X_train_texts,
        test_texts=X_test_texts,
        y_train=y_train,
        y_test=y_test,
        max_num_words=10000,
        max_sequence_length=100,
        batch_size=32,
        lstm_epochs=5,
        cnn_epochs=5,
        bert_epochs=2,
        bert_model_name='bert-base-uncased',
    )

    # Combine results for comparison
    all_results = ml_results + deep_results

    # 7. Display results
    print("\nEvaluation Results:")
    for result in all_results:
        print(f"{result.name}: Accuracy={result.accuracy:.4f}, Precision={result.precision:.4f}, "
              f"Recall={result.recall:.4f}, F1-Score={result.f1:.4f}")

    # 8. Plot comparison chart
    plot_results(all_results)


if __name__ == '__main__':
    # The main function is only called when the script is executed directly.
    main()