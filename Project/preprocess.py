"""
preprocess.py
=============

This module provides text cleaning, preprocessing, and balancing
functions.  Text messages are cleaned by lowercasing, removing
punctuation, numbers, and stopwords.  The dataset can also be
balanced via random under‑sampling.
"""

import re
from typing import List, Tuple

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import nltk
from nltk.corpus import stopwords


# Ensure stopwords are downloaded
nltk.download('stopwords')


def clean_text(text: str, stop_words: set) -> str:
    """
    Clean a single text string by lowercasing, removing non‑alphabetic
    characters, and stripping stopwords.

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
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)  # remove URLs
    text = re.sub(r'\S+@\S+', ' ', text)  # remove emails
    text = re.sub(r'\d+', ' ', text)  # remove numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove punctuation and special chars
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning to the entire DataFrame and return a new DataFrame
    with an additional column 'clean_text' containing cleaned text.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns 'text' and 'label'.

    Returns
    -------
    processed_df : pandas.DataFrame
        DataFrame with added 'clean_text' column.
    """
    english_stopwords = set(stopwords.words('english'))
    df['clean_text'] = df['text'].apply(lambda x: clean_text(x, english_stopwords))
    return df


def balance_dataset(
    texts: List[str], labels: List[int], random_state: int = 42
) -> Tuple[List[str], List[int]]:
    """
    Balance the dataset using random under‑sampling so that each class has the
    same number of samples.  Converts the input lists into a pandas
    DataFrame for compatibility with imbalanced‑learn API.

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
    X = pd.DataFrame({'text': texts})
    y = pd.Series(labels)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return X_resampled['text'].tolist(), y_resampled.tolist()
