# 📓 Notebook Guide — `AI-VS-HUMAN.ipynb`

A step-by-step walkthrough of the Jupyter notebook that trains the AI vs Human text detection model.

---

## Table of Contents

1. [Data Loading & Exploration](#1-data-loading--exploration)
2. [Text Preprocessing](#2-text-preprocessing)
3. [Feature Extraction (TF-IDF)](#3-feature-extraction-tf-idf)
4. [Train-Test Split](#4-train-test-split)
5. [Model 1: Logistic Regression](#5-model-1-logistic-regression)
6. [Model 2: Linear SVM](#6-model-2-linear-svm)
7. [Model Serialization](#7-model-serialization)

---

## 1. Data Loading & Exploration

**Cells:** 1–5

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('AI_Human.csv')
```

The dataset (`AI_Human.csv`) contains two columns:

| Column      | Description                                       |
| ----------- | ------------------------------------------------- |
| `text`      | The text sample (essay, paragraph, etc.)          |
| `generated` | Label — `0` for human-written, `1` for AI-generated |

**Exploration performed:**
- `df.head()` — Preview the first rows
- `df.info()` — Check data types and null counts
- `df.describe()` — Summary statistics
- `df['generated'].value_counts()` — Class distribution
- `sns.countplot(data=df, x='generated')` — Visualize the class balance

A new column `total` is also added to measure the character length of each text sample:
```python
df['total'] = df['text'].apply(lambda x: len(x))
```

---

## 2. Text Preprocessing

**Cells:** 6–15

The raw text goes through multiple cleaning stages before being fed to the model:

### 2.1 Remove Tags & Special Characters

```python
def remove_tags(text):
    tags = ['\n', '\'']
    for tag in tags:
        text = text.replace(tag, '')
    return text

df['text'] = df['text'].apply(remove_tags)
```

Strips newline characters and stray apostrophes from the text.

### 2.2 Remove Punctuation

```python
import string

def remove_punctuation(text):
    new_text = [x for x in text if x not in string.punctuation]
    new_text = ''.join(new_text)
    return new_text

df['text'] = df['text'].apply(remove_punctuation)
```

Removes all punctuation marks (`!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`).

### 2.3 Expand Contractions

```python
contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    # ... (full dictionary in notebook)
}

def cont_to_exp(x):
    if type(x) is str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x
    else:
        return x

df['text'] = df['text'].apply(lambda x: cont_to_exp(x))
```

Converts contractions to their expanded forms. This normalizes the text so the model doesn't treat "can't" and "cannot" as different features.

### 2.4 Remove Stopwords

```python
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as stopwords

df['text'] = df['text'].apply(
    lambda x: ' '.join([t for t in x.split() if t not in stopwords])
)
```

Removes common English stopwords (e.g., "the", "is", "and") using spaCy's built-in stop word list. This reduces noise and focuses the model on meaningful content words.

---

## 3. Feature Extraction (TF-IDF)

**Cells:** 16–18

```python
from sklearn.feature_extraction.text import TfidfVectorizer

X = df['text']
y = df['generated']

tfidf = TfidfVectorizer(norm='l1')
X = tfidf.fit_transform(X)
```

**What TF-IDF does:**
- **TF (Term Frequency)** — How often a word appears in a document
- **IDF (Inverse Document Frequency)** — Down-weights words that appear in many documents (common words contribute less)
- **L1 normalization** — Each document's feature vector sums to 1, preventing longer documents from dominating

The result is a sparse matrix where each row is a document and each column is a word from the learned vocabulary.

---

## 4. Train-Test Split

**Cell:** 19

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
```

- **80% training**, **20% testing**
- `random_state=42` ensures reproducibility
- `shuffle=True` randomizes the split

> **Note:** In the notebook, the test variables are named `X_text` and `y_test` (a minor typo — `X_text` should be `X_test`). This doesn't affect functionality.

---

## 5. Model 1: Logistic Regression

**Cells:** 20–24

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_lr(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(penalty='l2', C=1.0, tol=0.1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return clf
```

**Hyperparameters:**
| Parameter | Value | Meaning |
| --------- | ----- | ------- |
| `penalty` | `l2`  | Ridge regularization to prevent overfitting |
| `C`       | `1.0` | Inverse regularization strength (lower = stronger regularization) |
| `tol`     | `0.1` | Convergence tolerance |

The notebook tests this model on both AI-generated and human-written sample texts to verify predictions.

---

## 6. Model 2: Linear SVM ⭐ (Deployed Model)

**Cells:** 25–28

```python
from sklearn.svm import LinearSVC

def run_svm(X_train, y_train, X_test, y_test):
    clf = LinearSVC(penalty='l1', C=1.0, dual=False, loss='squared_hinge', tol=0.1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return clf
```

**Hyperparameters:**
| Parameter | Value            | Meaning |
| --------- | ---------------- | ------- |
| `penalty` | `l1`             | Lasso regularization — also performs implicit feature selection |
| `C`       | `1.0`            | Regularization strength |
| `dual`    | `False`          | Use primal formulation (required when `penalty='l1'`) |
| `loss`    | `squared_hinge`  | Squared hinge loss function |
| `tol`     | `0.1`            | Convergence tolerance |

> **Why SVM was chosen for deployment:** The Linear SVM with L1 penalty performs feature selection by zeroing out irrelevant word features, making it more robust for text classification. It was selected over Logistic Regression for the final deployment.

---

## 7. Model Serialization

**Cells:** 29–31

```python
import pickle

# Save the trained SVM model and its TF-IDF vectorizer
pickle.dump(clf_svm, open('clf.pkl', 'wb'))
pickle.dump(tfidf_svm, open('tfidf.pkl', 'wb'))
```

This exports two files:
- **`clf.pkl`** (~4 MB) — The trained Linear SVM classifier
- **`tfidf.pkl`** (~17 MB) — The fitted TF-IDF vectorizer (contains the full vocabulary)

These files are loaded by `main.py` at inference time to make predictions without retraining.

**Verification:**
```python
# Reload and verify the saved models work
svm_clf = pickle.load(open('clf.pkl', 'rb'))
svm_tfidf = pickle.load(open('tfidf.pkl', 'rb'))
```

---

## Summary

| Stage | Key Operation | Library |
| ----- | ------------- | ------- |
| Data Loading | `pd.read_csv()` | pandas |
| Visualization | `sns.countplot()` | seaborn |
| Preprocessing | Tag removal, punctuation, contractions, stopwords | string, NLTK, spaCy |
| Feature Extraction | `TfidfVectorizer(norm='l1')` | scikit-learn |
| Training | `LinearSVC(penalty='l1')` | scikit-learn |
| Serialization | `pickle.dump()` | pickle |
