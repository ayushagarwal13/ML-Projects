# Employee Attrition Prediction System

## Overview

Employee attrition is a major challenge for organizations because replacing employees involves recruitment costs, training expenses, and productivity loss.

This project aims to predict whether an employee is likely to leave the company based on factors such as age, income, job satisfaction, work-life balance, overtime, promotions, and other employee-related attributes.

The project demonstrates a complete Machine Learning workflow including:

* Data Understanding
* Exploratory Data Analysis (EDA)
* Data Preprocessing
* Feature Engineering
* Model Training
* Model Evaluation
* Feature Importance Analysis

---

## Business Problem

Organizations want to identify employees who are at risk of leaving so that HR teams can take proactive measures such as:

* Salary adjustments
* Promotions
* Career development opportunities
* Employee engagement programs

The goal is to predict:

**Attrition**

* Yes → Employee leaves the company
* No → Employee stays in the company

---

## Dataset Information

The dataset contains employee-related information including:

| Feature                    | Description                |
| -------------------------- | -------------------------- |
| Age                        | Employee age               |
| Gender                     | Male/Female                |
| Marital_Status             | Marital status of employee |
| Department                 | Department of employee     |
| Job_Role                   | Employee role              |
| Monthly_Income             | Monthly salary             |
| Years_at_Company           | Total years in company     |
| Years_Since_Last_Promotion | Years since last promotion |
| Job_Satisfaction           | Satisfaction rating (1–5)  |
| Work_Life_Balance          | Work-life balance rating   |
| Overtime                   | Overtime status            |
| Attrition                  | Target Variable            |

Dataset Size:

* 1000 Records
* 26 Columns

---

## Project Workflow

### 1. Data Loading

The dataset was loaded using Pandas and inspected using:

```python
df.head()
df.info()
```

---

### 2. Data Cleaning

Checked for missing values:

```python
df.isnull().sum()
```

Observation:

* No missing values were found.

---

### 3. Exploratory Data Analysis (EDA)

EDA was performed to understand relationships between features and employee attrition.

Examples:

* Overtime vs Attrition
* Job Satisfaction vs Attrition
* Monthly Income vs Attrition
* Promotion History vs Attrition

Example:

```python
pd.crosstab(
    df["Overtime"],
    df["Attrition"],
    normalize="index"
) * 100
```

---

### 4. Feature Selection

Employee_ID was removed because it is a unique identifier and does not contribute to prediction.

```python
df = df.drop("Employee_ID", axis=1)
```

---

### 5. Feature and Target Separation

```python
X = df.drop("Attrition", axis=1)
y = df["Attrition"]
```

Where:

* X = Input Features
* y = Target Variable

---

### 6. Encoding Categorical Variables

Target Encoding:

```python
y = y.map({"No":0,"Yes":1})
```

One-Hot Encoding:

```python
X = pd.get_dummies(X, drop_first=True)
```

Reason:

* Machine Learning algorithms require numerical inputs.
* One-Hot Encoding avoids introducing artificial hierarchy.

---

### 7. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Split Ratio:

* Training Data: 80%
* Testing Data: 20%

---

## Models Used

### Logistic Regression

Baseline classification model:

```python
from sklearn.linear_model import LogisticRegression
```

---

### Balanced Logistic Regression

Used class balancing to handle dataset imbalance:

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)
```

---

### Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier
```

Used to capture complex feature interactions and non-linear relationships.

---

## Evaluation Metrics

The following metrics were used:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Example:

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

---

## Key Learning

One of the most important findings from this project was that:

**High Accuracy does not always mean a good model.**

Initial Logistic Regression achieved:

* Accuracy ≈ 81%

However, the confusion matrix revealed that the model predicted only the majority class and failed to identify employees who left.

After applying:

```python
class_weight="balanced"
```

Recall improved significantly, making the model more useful for attrition prediction.

---

## Feature Importance

Random Forest identified the following important features:

* Monthly Income
* Training Hours Last Year
* Hourly Rate
* Distance From Home
* Age
* Years at Company
* Absenteeism

Feature importance helps understand which factors contribute most to model predictions.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Google Colab

---

## Future Improvements

Possible enhancements:

* Hyperparameter Tuning
* Cross Validation
* SMOTE for handling class imbalance
* XGBoost Classifier
* Deployment using Streamlit
* Real-time employee attrition prediction dashboard

---

## Conclusion

This project demonstrates a complete Machine Learning pipeline for employee attrition prediction.

The project covers data preprocessing, exploratory data analysis, model training, evaluation, and interpretation of results. It also highlights the importance of choosing appropriate evaluation metrics such as Recall in imbalanced classification problems.
