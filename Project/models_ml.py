"""
models_ml.py
============

This module defines functions for vectorizing text using TF‑IDF and
training traditional machine learning models for text classification.
The models include Logistic Regression, Linear SVM and Random Forest.
"""

from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .evaluation import ModelResult


def vectorize_texts(train_texts: List[str], test_texts: List[str]) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    Convert lists of text messages into TF‑IDF feature matrices for training
    and testing.

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
        The fitted vectorizer.
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
    Train traditional ML models and evaluate them on the test set.  The
    models include Logistic Regression, Linear SVM and Random Forest.

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
        Evaluation results for each model.
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
