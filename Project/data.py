"""
data.py
=======

This module contains functionality for downloading and sampling the
SMS Spam Collection dataset.  The dataset is loaded via the
`datasets` library and converted into a pandas DataFrame.  A utility
function allows sampling a specified number of rows from the full
dataset to satisfy project size constraints.
"""

import pandas as pd
from datasets import load_dataset


def download_and_sample_dataset(seed: int = 42, sample_size: int = 5000) -> pd.DataFrame:
    """
    Download the SMS Spam Collection dataset and return a pandas DataFrame
    limited to `sample_size` samples.  If the dataset contains fewer
    samples than `sample_size`, the full dataset is returned.

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
    # Load dataset using huggingface datasets library
    dataset = load_dataset('ucirvine/sms_spam', split='train')
    df = pd.DataFrame(dataset)
    # Rename column for convenience
    df = df.rename(columns={'sms': 'text'})
    # Sample if larger than allowed
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return df
