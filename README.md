# SMS Classification (NLP_MCD)

This project implements a **full text‑classification pipeline** to identify SMS messages as `ham` (legitimate) or `spam`. Compared with earlier versions, the folder structure and the set of available deep models have been updated. The code downloads and prepares the dataset, balances the classes, trains classical and lightweight deep‑learning models, compares their performance, and generates visualisations of the results.

## Repository structure

```
NLP_MCD/
├── Project/              # Package with the source code
│   ├── __init__.py
│   ├── data.py          # Downloads and samples the SMS Spam Collection dataset
│   ├── preprocess.py    # Text cleaning and balancing via random under‑sampling
│   ├── models_ml.py     # TF–IDF vectorisation and classical models (logistic regression, SVM, random forest)
│   ├── models_dl.py     # Deep‑learning models (LSTM, LSTM+GloVe, CNN, compact Transformer)
│   ├── evaluation.py    # Container for results and plotting function
│   ├── main.py          # Entry point that runs the entire pipeline
│   └── __init__.py
├── Notebook/            # Jupyter notebook and its HTML/PDF export
│   ├── Project_Notebook.ipynb
│   ├── Project_Notebook.html
│   └── Project_Notebook.pdf
├── Doc/                 # Scientific paper (PDF/LaTeX) with analysis and discussion
│   ├── Comparative_Evaluation_of_Traditional_and_Deep_Learning_Models_for_SMS_Spam_Detection.pdf
│   ├── main.tex
│   ├── references.bib
│   ├── fig_balanced.png
│   ├── fig_mean_len.png
│   └── fig_model_perf.png
└── README.md
```

## Installation and dependencies

1. Ensure you have Python 3.7 or later.
2. Install the required dependencies by running:

   ```bash
   pip install pandas numpy scikit-learn nltk imbalanced-learn matplotlib seaborn datasets tensorflow==2.* gensim
   ```

   The `transformers` package is no longer required: the current version replaces the BERT model with a **compact Transformer** implemented in Keras.

3. On first run the preprocessing step automatically downloads NLTK stopwords.

## Running the pipeline

To train and evaluate all models, run the main script from the project root:

```bash
python -m Project.main
```

The workflow proceeds through the following stages:

1. **Data selection.** The **SMS Spam Collection** corpus (5 574 messages) is downloaded and a random sample of 5 000 messages is drawn to reduce the size of the experiment【899904514028581†L270-L283】.
2. **Pre‑processing.** Messages are lower‑cased, URLs, email addresses, numbers and punctuation are removed, they are tokenised and stopwords are filtered out.
3. **Dataset balancing.** Since only around 13 % of the messages are spam, the majority class (ham) is randomly undersampled until the classes are equal. This yields ≈650 ham and 650 spam messages (1 324 examples in total)【899904514028581†L270-L303】.
4. **Train–test split.** The balanced set is split into training and test sets (80/20) with stratification.
5. **Classical models.** Texts are vectorised using TF–IDF (unigrams and bigrams) and three classifiers are trained: *logistic regression*, *linear SVM* and *random forest*【899904514028581†L270-L284】.
6. **Deep‑learning models.** Using fixed‑length padded sequences of word tokens, four lightweight neural networks are trained:
   - **LSTM:** a recurrent network with randomly initialised embeddings.
   - **LSTM + GloVe:** the same architecture, but embeddings are initialised with 100‑dimensional GloVe vectors (kept frozen).
   - **CNN:** a 1‑D convolutional network inspired by Kim (2014).
   - **Compact Transformer:** a self‑contained architecture with multi‑head attention, normalisation and global averaging; it replaces the BERT model used originally.

7. **Evaluation.** Accuracy, precision, recall and F1 are computed on the test set for each model and a summary is printed. A bar chart (`model_performance_comparison.png`) comparing all metrics is also generated.

## Key results

On this reduced and balanced dataset, the linear classifiers outperform the deep networks. In particular, **logistic regression** and the **linear SVM** achieve about 95 % accuracy and F1≈0.954【899904514028581†L270-L284】. The **random forest** trails slightly with 93 % accuracy. Both **LSTM** variants (with or without GloVe) converge to predicting every message as spam, achieving 100 % recall but low precision and accuracy of ≈0.50【899904514028581†L270-L284】. The **CNN** offers a trade‑off: 84.9 % accuracy and high recall (98 %), but precision of only 77 %. The **compact Transformer** obtains 63 % accuracy with perfect recall but many false positives【899904514028581†L270-L284】.

These findings align with recent literature: when large corpora and pre‑trained models are available, transformer and hybrid approaches can exceed 97–99 % accuracy【899904514028581†L321-L325】. However, under data and computational constraints, linear methods with TF–IDF remain a robust and efficient choice【899904514028581†L332-L349】.

## Additional resources

* **Notebook:** The `Notebook/` directory contains a Jupyter notebook with the entire workflow, including code cells, visualisations and exploratory analysis.
* **Paper:** The `Doc/` folder contains a paper in PDF/LaTeX summarising the motivation, methodology, results and discussion of the project. The figures `fig_balanced.png`, `fig_mean_len.png` and `fig_model_perf.png` illustrate the class balance, the mean message length by class and the metric comparison respectively.

## Licence

The original dataset (SMS Spam Collection) is licensed under CC BY 4.0. The code in this repository is provided for educational purposes and may be reused with attribution to the authors.