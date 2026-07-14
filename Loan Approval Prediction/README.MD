# 🏦 Loan Approval Prediction using Machine Learning

An end-to-end Machine Learning project that predicts whether a loan application will be **Approved** or **Rejected** based on an applicant's financial and personal information. The project covers the complete machine learning workflow, including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and feature importance analysis.

---

## 📌 Problem Statement

Financial institutions receive thousands of loan applications every day. Manually evaluating each application is time-consuming and prone to inconsistencies. The objective of this project is to build a machine learning model that can automatically predict whether a loan application should be approved based on applicant details such as income, loan amount, CIBIL score, assets, education, and employment status.

---

## 📂 Dataset

- **Source:** Kaggle
- **Dataset:** Loan Approval Prediction Dataset

### Features

- Loan ID
- Number of Dependents
- Education
- Self Employed
- Annual Income
- Loan Amount
- Loan Term
- CIBIL Score
- Residential Assets Value
- Commercial Assets Value
- Luxury Assets Value
- Bank Asset Value
- Loan Status (Target Variable)

---

# 🚀 Project Workflow

### 1️⃣ Import Required Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

### 2️⃣ Load the Dataset

- Read the dataset into a Pandas DataFrame
- Inspect the data
- Clean column names

---

### 3️⃣ Data Understanding

- Dataset Shape
- Dataset Information
- Statistical Summary
- Column Analysis

---

### 4️⃣ Data Quality Checks

- Missing Values
- Duplicate Records
- Remove unnecessary columns
- Target Variable Analysis

---

### 5️⃣ Exploratory Data Analysis (EDA)

Performed exploratory analysis using various visualizations:

- Loan Status Distribution
- Education Distribution
- Self Employment Distribution
- Income Distribution
- Correlation Heatmap
- Feature-wise Analysis

---

### 6️⃣ Data Preprocessing

- Label Encoding
- Feature Selection
- Train-Test Split

---

### 7️⃣ Machine Learning Models

The following classification models were trained and evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

---

### 8️⃣ Model Evaluation

Models were evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1 Score

---

### 9️⃣ Feature Importance

Random Forest feature importance was analyzed to identify the most influential features affecting loan approval decisions.

---

# 📊 Model Performance

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 72.72% |
| Decision Tree | 97.78% |
| Random Forest | 97.78% |

**Best Performing Model:** Random Forest Classifier

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

---

# 📁 Project Structure

```
Loan-Approval-Prediction/
│
├── Loan_Approval_Prediction.ipynb
├── loan_approval_dataset.csv
├── loan_approval_model.pkl
├── README.md
└── requirements.txt
```

---

# 📈 Key Learnings

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Correlation Analysis
- Label Encoding
- Feature Engineering
- Train-Test Split
- Machine Learning Model Training
- Model Evaluation
- Feature Importance Analysis

---

# 🔮 Future Improvements

- Hyperparameter Tuning using GridSearchCV
- Cross Validation
- Model Deployment using Streamlit
- Docker Deployment
- Cloud Deployment

---

# ▶️ How to Run

### Clone the repository

```bash
git clone https://github.com/ayushagarwal13/Loan-Approval-Prediction.git
```

### Navigate to the project directory

```bash
cd Loan-Approval-Prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the notebook

Open:

```
Loan_Approval_Prediction.ipynb
```

using Google Colab or Jupyter Notebook.

---

# 📚 Libraries Used

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

---

# ⭐ If you found this project useful

Please consider giving this repository a ⭐ on GitHub.

---

## 👨‍💻 Author

**Ayush Agarwal**

Software Engineer | AI & Machine Learning Enthusiast

GitHub: https://github.com/ayushagarwal13

LinkedIn: *Add your LinkedIn profile here*

Email: *ayush.techmail@gmail.com*
