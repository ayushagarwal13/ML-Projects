# 🌸 Iris Flower Classification using K-Nearest Neighbors (KNN)

A beginner-friendly yet comprehensive machine learning project that classifies Iris flowers into three distinct species — *Iris Setosa*, *Iris Versicolor*, and *Iris Virginica* — using the **K-Nearest Neighbors (KNN)** classification algorithm. This project covers the full ML pipeline: from data loading and exploratory analysis to model training and performance evaluation.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Model & Evaluation](#-model--evaluation)
- [Project Structure](#-project-structure)
- [Key Concepts](#-key-concepts)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🔍 Overview

The **Iris dataset** is one of the most well-known datasets in the machine learning community, originally introduced by British statistician Ronald Fisher in 1936. It is commonly used as a benchmark for classification algorithms.

In this project, we:
- Perform **Exploratory Data Analysis (EDA)** to understand the dataset
- Preprocess the data by separating features and labels
- Train a **KNN classifier** to predict the species of an Iris flower based on its physical measurements
- Evaluate the model using multiple metrics including accuracy, confusion matrix, and a detailed classification report

---

## 📁 Dataset

- **Source:** [Iris Flower Dataset on Kaggle](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)
- **File:** `IRIS.csv`
- **Total Samples:** 150 rows (perfectly balanced — 50 samples per species)

### Features (Input Variables)

| Feature | Description | Unit |
|---|---|---|
| `sepal_length` | Length of the sepal | cm |
| `sepal_width` | Width of the sepal | cm |
| `petal_length` | Length of the petal | cm |
| `petal_width` | Width of the petal | cm |

### Target (Output Variable)

| Class | Species |
|---|---|
| 0 | Iris-setosa |
| 1 | Iris-versicolor |
| 2 | Iris-virginica |

> **Note:** Iris-setosa is linearly separable from the other two species, making it the easiest to classify. Versicolor and Virginica have some overlap, making them slightly harder to distinguish.

---

## 🧪 Project Workflow

### Step 1 — Load Dataset
The dataset is downloaded from Kaggle using the `kagglehub` library and loaded into a pandas DataFrame for easy manipulation.

```python
import kagglehub
path = kagglehub.dataset_download("arshid/iris-flower-dataset")

import pandas as pd
df = pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv")
```

### Step 2 — Exploratory Data Analysis (EDA)
Before training any model, we explore the dataset to understand its structure, detect any anomalies, and get a feel for the data distribution.

- `df.shape` — Confirms 150 rows and 5 columns
- `df.info()` — Checks data types and null values (none found)
- `df['species'].value_counts()` — Confirms balanced class distribution (50 each)
- `df.describe()` — Summary statistics (mean, std, min, max, quartiles)

### Step 3 — Feature & Target Separation
We separate the independent variables (features) from the dependent variable (target label).

```python
X = df.drop("species", axis=1)   # Features: sepal/petal length & width
y = df["species"]                 # Target: species name
```

### Step 4 — Train-Test Split
The dataset is split into a **training set (80%)** and a **testing set (20%)** using scikit-learn's `train_test_split`. A fixed `random_state=42` ensures reproducibility.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Results in: 120 training samples, 30 testing samples
```

### Step 5 — Model Training (KNN)
A **K-Nearest Neighbors** classifier is instantiated with `k=3` (3 nearest neighbors) and trained on the training data.

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

### Step 6 — Model Evaluation
The trained model is evaluated using three complementary metrics:

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `Python 3` | Core programming language |
| `pandas` | Data loading, manipulation, and EDA |
| `scikit-learn` | ML model, train-test split, and evaluation metrics |
| `kagglehub` | Downloading dataset directly from Kaggle |

---

## 🚀 Getting Started

### Prerequisites
Make sure you have **Python 3.7+** installed. A Kaggle account is required to download the dataset.

### 1. Clone the repository

```bash
git clone https://github.com/ayushagarwal13/iris-flower-classification.git
cd iris-flower-classification
```

### 2. Install dependencies

```bash
pip install pandas scikit-learn kagglehub
```

### 3. Set up Kaggle API credentials
To use `kagglehub`, configure your Kaggle API key:
1. Go to [kaggle.com](https://www.kaggle.com) → Account → Create API Token
2. Place the downloaded `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<user>\.kaggle\` (Windows)

### 4. Run the notebook
Open the notebook in Jupyter or Google Colab and run all cells sequentially:

```bash
jupyter notebook Iris_Flower_Classification.ipynb
```

Or simply open it directly in [Google Colab](https://colab.research.google.com/).

---

## 📊 Model & Evaluation

### Algorithm: K-Nearest Neighbors (KNN)
KNN is a simple, non-parametric, instance-based learning algorithm. It classifies a data point based on the majority class among its `k` nearest neighbors in the feature space, using Euclidean distance by default.

- **`k = 3`** was chosen for this project
- No explicit training phase — the model memorizes the training data
- Works well on small, well-structured datasets like Iris

### Expected Results

The KNN model with `k=3` typically achieves around **~97% accuracy** on the Iris test set.

**Sample Confusion Matrix:**
```
[[10  0  0]
 [ 0  9  1]
 [ 0  0 10]]
```

**Sample Classification Report:**
```
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        10
Iris-versicolor       1.00      0.90      0.95        10
 Iris-virginica       0.91      1.00      0.95        10

       accuracy                           0.97        30
      macro avg       0.97      0.97      0.97        30
   weighted avg       0.97      0.97      0.97        30
```

> Actual results may vary slightly depending on the random state and data split.

---

## 📂 Project Structure

```
iris-flower-classification/
│
├── Iris_Flower_Classification.ipynb   # Main Jupyter notebook (full pipeline)
├── README.md                          # Project documentation (this file)
└── LICENSE                            # MIT License
```

---

## 💡 Key Concepts

- **KNN (K-Nearest Neighbors):** A lazy learning algorithm that classifies based on proximity to training examples in feature space.
- **Train-Test Split:** Dividing data into separate subsets to train and evaluate a model without data leakage.
- **Accuracy Score:** The proportion of correctly classified samples out of total samples.
- **Confusion Matrix:** A table showing true vs. predicted labels — useful for identifying which classes are being misclassified.
- **Classification Report:** Per-class breakdown of precision, recall, and F1-score.

---

## 🔮 Future Improvements

- Add **feature scaling** (StandardScaler) to potentially improve KNN performance
- Experiment with different values of **k** and plot an elbow curve to find the optimal number
- Try other classification algorithms: **Decision Tree**, **SVM**, **Random Forest**, **Logistic Regression**
- Add **data visualizations**: pair plots, heatmaps, and decision boundary plots using `seaborn` and `matplotlib`
- Build a simple **web app** using Streamlit to take user inputs and predict the Iris species in real time

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

---

> ⭐ If you found this project helpful, consider giving it a star on GitHub!

