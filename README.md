Repo Name
clothing-review-nlp-classification

Description
NLP pipeline for classifying clothing review recommendations. Covers text preprocessing, bag-of-words and word embedding feature representations (TF-IDF weighted/unweighted), and ML classification models with 5-fold cross-validation. 
README
markdown# 👗 Clothing Review NLP Classification

An end-to-end NLP pipeline built in Python for predicting whether a clothing
review recommends a product. Completed as Assignment 3 — Milestone 1 for
COSC2820 (Advanced Programming for Data Science) at RMIT University.

**Data source:** [Women's E-Commerce Clothing Reviews (Kaggle)](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
*(Note: Dataset modified for course use)*

## 📋 Project Overview

| Task | Description | Output |
|---|---|---|
| Task 1 | Text preprocessing pipeline | `vocab.txt`, `processed.csv` |
| Task 2 | Feature representation generation | `count_vectors.txt` |
| Task 3 | ML classification & evaluation | Results in `task2_3.ipynb` |

## 🔧 Tasks

### Task 1 — Basic Text Pre-processing (`task1.ipynb`)

Applied to the **Review Text** column (~19,600 reviews):

- Tokenization using regex `r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"`
- Lowercasing all tokens
- Removal of words with length < 2
- Stop word removal using `stopwords_en.txt`
- Removal of words appearing only once (term frequency)
- Removal of top 20 most frequent words (document frequency)
- Output: `processed.csv` and `vocab.txt` (alphabetically sorted, zero-indexed)

### Task 2 — Feature Representations (`task2_3.ipynb`)

Three feature representations built from **Review Text** only:

- **Bag-of-words:** sparse count vector based on `vocab.txt`
- **Unweighted word embeddings:** average word vectors using a pretrained model (FastText/GloVe/Word2Vec)
- **TF-IDF weighted word embeddings:** weighted average using TF-IDF scores

### Task 3 — Review Classification (`task2_3.ipynb`)

Two experiments using 5-fold cross-validation:

**Q1 — Language model comparison:** Which feature representation (BoW, unweighted, TF-IDF weighted) performs best?

**Q2 — Does more information help?** Comparing models trained on:
- Title only
- Review Text only
- Title + Review Text combined

Models evaluated include at minimum: bag-of-words classifier, weighted embedding classifier, and unweighted embedding classifier.

## 🗂️ Repository Structure


s3859590/
├── task1.ipynb                  # Task 1: Text preprocessing
├── task1.py                     # .py export of task1.ipynb
├── task2_3.ipynb                # Tasks 2 & 3: Features + classification
├── task2_3.py                   # .py export of task2_3.ipynb
├── vocab.txt                    # Vocabulary (word:index, alphabetical)
├── count_vectors.txt            # Sparse count vector representations
├── processed.csv                # Cleaned review dataset
└── stopwords_en.txt             # Stop words list (provided)


## 🛠️ Libraries Used

```python
import re
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import gensim  # or equivalent for word embeddings
```

## 👩‍💻 Author

**Prathibha Magesh** — RMIT University, s3859590
