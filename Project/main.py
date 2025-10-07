"""
main.py
=======

Entry point for the text classification project.  This script ties
together the data loading, preprocessing, model training and
evaluation steps.  To run the full pipeline, execute this file
directly:

```bash
python -m text_classification_project.main
```

The script will print evaluation metrics to stdout and save a bar
chart comparing model performance as `model_performance_comparison.png`.
"""

from sklearn.model_selection import train_test_split

from .data import download_and_sample_dataset
from .preprocess import preprocess_dataframe, balance_dataset
from .models_ml import vectorize_texts, train_ml_models
from .models_dl import train_deep_models
from .evaluation import plot_results


def run_pipeline() -> None:
    """
    Execute the full workflow: download and preprocess the data, balance
    it, train models, evaluate them, and generate comparison plots.
    Results are printed to the console and a bar chart is saved to disk.
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
    X_train_tfidf, X_test_tfidf, _ = vectorize_texts(X_train_texts, X_test_texts)
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

    # Combine results
    all_results = ml_results + deep_results

    # 7. Display results
    print("\nEvaluation Results:")
    for result in all_results:
        print(f"{result.name}: Accuracy={result.accuracy:.4f}, "
              f"Precision={result.precision:.4f}, Recall={result.recall:.4f}, "
              f"F1-Score={result.f1:.4f}")

    # 8. Plot comparison chart
    plot_results(all_results)


def main():
    """
    Wrapper for executing the pipeline when running as a script.
    """
    run_pipeline()


if __name__ == '__main__':
    main()