# 🚗 Car Price Prediction using Machine Learning

## 📌 Project Overview

This project predicts the selling price of a used car based on various features such as present price, kilometers driven, fuel type, transmission type, owner count, and car age.

The project demonstrates the complete Machine Learning workflow including data preprocessing, feature engineering, model training, evaluation, and prediction using real-world car data.

---

## 🎯 Objective

The objective of this project is to build a machine learning model that can accurately estimate the selling price of a used car.

This helps buyers and sellers determine a fair market value for a vehicle based on its specifications and condition.

---

## 📂 Dataset

Dataset Source:

Car Price Prediction Dataset from Kaggle

### Features

* Car_Name
* Year
* Present_Price
* Kms_Driven
* Fuel_Type
* Seller_Type
* Transmission
* Owner

### Target Variable

* Selling_Price

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Google Colab / Jupyter Notebook

---

## 🔍 Project Workflow

### 1. Data Collection

* Loaded the dataset using Pandas.

### 2. Data Exploration

* Checked dataset shape.
* Inspected data types.
* Analyzed statistical information.
* Checked missing values.

### 3. Data Preprocessing

* Removed the `Car_Name` column.
* Created a new feature called `Car_Age`.
* Converted categorical variables into numerical format using One-Hot Encoding.

### 4. Feature Engineering

Created:

```python
Car_Age = 2026 - Year
```

Dropped the original `Year` column after creating `Car_Age`.

### 5. Train-Test Split

```python
train_test_split(
    test_size=0.2,
    random_state=42
)
```

* Training Data: 240 records
* Testing Data: 61 records

### 6. Model Building

Implemented:

* Linear Regression
* Random Forest Regressor

### 7. Model Evaluation

Metrics used:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score

---

## 📊 Results

### Random Forest Regressor

| Metric   | Value  |
| -------- | ------ |
| MAE      | 0.6387 |
| MSE      | 0.9216 |
| R² Score | 0.9600 |

### Linear Regression

| Metric   | Value  |
| -------- | ------ |
| R² Score | 0.8490 |

### Best Model

✅ Random Forest Regressor

Random Forest outperformed Linear Regression and achieved an R² score of approximately 96%.

---

## 📈 Feature Importance

| Feature                | Importance |
| ---------------------- | ---------- |
| Present_Price          | 0.8829     |
| Car_Age                | 0.0601     |
| Kms_Driven             | 0.0362     |
| Transmission_Manual    | 0.0104     |
| Fuel_Type_Petrol       | 0.0040     |
| Fuel_Type_Diesel       | 0.0029     |
| Seller_Type_Individual | 0.0021     |
| Owner                  | 0.0014     |

### Key Insight

The showroom price (`Present_Price`) was the most influential factor affecting the predicted selling price.

---

## 🚀 Sample Prediction

Input:

```python
[5.59, 27000, 0, 12, 0, 1, 0, 1]
```

Predicted Selling Price:

```python
₹3.82 Lakh
```

---

## 📚 Key Learnings

* Exploratory Data Analysis (EDA)
* Feature Engineering
* One-Hot Encoding
* Regression Algorithms
* Model Evaluation
* Feature Importance Analysis
* Real-World Machine Learning Workflow

---

## 👨‍💻 Author

Ayush Agarwal

If you found this project useful, feel free to ⭐ the repository.
