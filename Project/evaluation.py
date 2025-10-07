"""
evaluation.py
=============

This module defines a dataclass for storing evaluation metrics and
provides a utility function to plot model performance comparisons.
"""

from dataclasses import dataclass
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ModelResult:
    """Simple container for model evaluation results."""
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float


def plot_results(results: List[ModelResult], title: str = 'Model Performance Comparison') -> None:
    """
    Plot bar charts comparing accuracy, precision, recall and F1‑score for
    multiple models.  The plot is saved as 'model_performance_comparison.png'
    in the current working directory.

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
