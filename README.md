# 💼 Employee Salary Prediction

## 📌 Project Overview

**Employee Salary Prediction** is a Machine Learning classification project that predicts whether an individual's annual income is:

* **>50K**
* **<=50K**

based on demographic, educational, and employment-related attributes.

The project uses the **Adult Income dataset** and follows an end-to-end Machine Learning workflow including data cleaning, exploratory data analysis, feature engineering, feature encoding, feature scaling, model training, model comparison, hyperparameter tuning, and model saving.

The final trained model can be used through a **Streamlit frontend** where users enter employee information and receive an income prediction.

---

## 🎯 Problem Statement

The objective of this project is to build a Machine Learning model that can classify an individual's annual income as **greater than $50,000 or less than or equal to $50,000**.

The model uses historical demographic, educational, and employment-related information to make the prediction.

---

## 📊 Dataset

The project uses the **Adult Income dataset**.

The original dataset contains:

* **32,561 records**
* **15 columns**
* **6 numerical features**
* **9 categorical features**

### Dataset Features

| Feature          | Description                           |
| ---------------- | ------------------------------------- |
| `age`            | Age of the individual                 |
| `workclass`      | Type of employment                    |
| `fnlwgt`         | Final weight                          |
| `education`      | Education level                       |
| `education.num`  | Numerical representation of education |
| `marital.status` | Marital status                        |
| `occupation`     | Occupation                            |
| `relationship`   | Relationship status                   |
| `race`           | Race                                  |
| `sex`            | Gender                                |
| `capital.gain`   | Capital gain                          |
| `capital.loss`   | Capital loss                          |
| `hours.per.week` | Hours worked per week                 |
| `native.country` | Country of origin                     |
| `income`         | Target variable                       |

The notebook removes `education.num` and `fnlwgt` during feature selection.

---

# 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Inspection
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Feature Selection
   ↓
Categorical Encoding
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Final Model Selection
   ↓
Model Saving
   ↓
Streamlit Frontend
   ↓
Income Prediction
```

---

# 🧹 Data Cleaning

The dataset was inspected for:

* Missing values
* Duplicate records
* Data types
* Statistical information
* Categorical values
* Numerical distributions

The dataset contained **24 duplicate rows**, which were removed during preprocessing.

The dataset uses `?` values in some categorical columns to represent missing/unknown values.

---

# 🔎 Exploratory Data Analysis

Exploratory Data Analysis was performed using:

* Pandas
* NumPy
* Matplotlib
* Seaborn

The analysis included:

* Univariate analysis
* Distribution analysis
* Categorical feature analysis
* Income distribution
* Relationship between age and income
* Relationship between education and income
* Relationship between working hours and income
* Outlier detection
* Correlation analysis

---

# ⚙️ Feature Engineering

A new feature called:

```text
total.capital
```

was created using:

```python
total.capital = capital.gain - capital.loss
```

This feature was introduced to represent the combined effect of capital gain and capital loss.

The notebook also removed:

```text
education.num
fnlwgt
```

during feature selection.

---

# 🔤 Feature Encoding

Different encoding techniques were used for different types of categorical variables.

### Label Encoding

Label encoding was applied to:

* `income`
* `sex`

### Ordinal Encoding

`education` was encoded using an ordinal ordering of education levels.

### One-Hot Encoding

One-hot encoding was applied to:

* `workclass`
* `marital.status`
* `occupation`
* `relationship`
* `race`
* `native.country`

This resulted in the categorical variables being converted into numerical features that could be used by the Machine Learning models.

---

# ✂️ Train-Test Split

The dataset was divided into training and testing sets using:

```python
train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42
)
```

Therefore:

* **67%** data → Training
* **33%** data → Testing

---

# 📏 Feature Scaling

`StandardScaler` was used for the following numerical features:

```text
age
education
capital.gain
capital.loss
hours.per.week
```

The scaler was fitted on the training data and then used to transform both training and testing data.

The scaler was also saved separately as:

```text
scaler_employee.joblib
```

---

# 🤖 Machine Learning Models

Multiple classification algorithms were trained and compared.

The models included:

* Logistic Regression
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* Support Vector Machine
* XGBoost
* LightGBM
* CatBoost

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

---

# 📊 Model Comparison

The initial model comparison produced the following results:

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     84.18% |     69.33% |     59.61% |     64.10% |     0.8934 |
| KNN                 |     85.89% |     71.57% |     67.15% |     69.29% |     0.8911 |
| Decision Tree       |     82.35% |     62.95% |     62.08% |     62.51% |     0.7698 |
| Random Forest       |     85.31% |     71.86% |     62.51% |     66.86% |     0.8992 |
| SVM                 |     80.59% |     74.04% |     27.90% |     40.53% |     0.8664 |
| XGBoost             |     87.20% |     76.38% |     66.56% |     71.13% |     0.9262 |
| **LightGBM**        | **87.42%** | **77.29%** | **66.44%** | **71.46%** | **0.9254** |
| CatBoost            |     87.38% |     77.12% |     66.48% |     71.41% |     0.9263 |

LightGBM gave the strongest overall initial performance and was selected for further hyperparameter tuning.

---

# 🎯 Hyperparameter Tuning

`GridSearchCV` was used to tune the LightGBM model.

The parameter grid included:

```python
{
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "n_estimators": [100, 200],
    "num_leaves": [15, 31]
}
```

The best LightGBM configuration obtained was:

```text
learning_rate = 0.1
max_depth     = 5
n_estimators  = 200
num_leaves    = 15
```

## The tuned LightGBM model was selected as the final model.

# 🏆 Final Model Performance

The final tuned LightGBM model achieved:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **87.37%** |
| Precision | **77.61%** |
| Recall    | **65.66%** |
| F1 Score  | **71.14%** |
| ROC-AUC   | **0.9268** |

These results are based on the test set used in the notebook.

---

# ⭐ Why LightGBM?

LightGBM was selected as the final model because it provided strong classification performance compared with the other models tested.

It achieved a high ROC-AUC score while maintaining a good balance between precision and recall.

---

# 💾 Model Saving

The final trained model was saved using `joblib`.

```python
import joblib

joblib.dump(final_best_model, "employee_salary_model.joblib")
joblib.dump(scaler, "scaler_employee.joblib")
```

Therefore, the project contains:

```text
employee_salary_model.joblib
scaler_employee.joblib
```

These files can be loaded by the frontend to make predictions without retraining the model.

---

# 🌐 Streamlit Frontend

A Streamlit frontend can be used to provide an interactive interface for the trained model.

The frontend should collect the same input features used by the model.

## 📝 User Inputs

The prediction form should contain:

### Personal Information

* Age
* Sex
* Race
* Relationship
* Marital Status
* Native Country

### Employment Information

* Workclass
* Occupation
* Hours per Week

### Education Information

* Education Level

### Financial Information

* Capital Gain
* Capital Loss

The project also creates `total.capital` from capital gain and capital loss, so the frontend does **not need to ask the user separately for `total.capital`**. It can be calculated automatically:

```python
total_capital = capital_gain - capital_loss
```

The notebook's prediction example uses these same input fields and derives the required processed input before prediction.

---

# 🖥️ Suggested Frontend Layout

A clean Streamlit interface can be structured as:

```text
------------------------------------------------
        💼 Employee Salary Prediction
------------------------------------------------

Enter Employee Information

Personal Information
--------------------------------
Age             [       ]
Sex             [ Select ]
Race            [ Select ]
Relationship    [ Select ]

Employment Information
--------------------------------
Workclass       [ Select ]
Occupation      [ Select ]
Hours/Week      [       ]

Education
--------------------------------
Education Level [ Select ]

Financial Information
--------------------------------
Capital Gain    [       ]
Capital Loss    [       ]

              [ Predict Income ]
------------------------------------------------

Prediction:
🟢 Income > $50K

or

🔴 Income <= $50K
------------------------------------------------
```

---

# 🔁 Streamlit Prediction Workflow

```text
User enters information
        ↓
Streamlit collects input
        ↓
Create Pandas DataFrame
        ↓
Create total.capital
        ↓
Apply same encoding
        ↓
Apply saved scaler
        ↓
Load employee_salary_model.joblib
        ↓
Model prediction
        ↓
Display income category
```

---

# 📁 Recommended Project Structure

```text
Employee-Salary-Prediction/
│
├── data/
│   └── adult.csv
│
├── notebook/
│   └── Employee-Salary-Prediction.ipynb
│
├── model/
│   ├── employee_salary_model.joblib
│   └── scaler_employee.joblib
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🛠️ Technologies Used

| Technology       | Purpose                    |
| ---------------- | -------------------------- |
| Python           | Programming                |
| Pandas           | Data manipulation          |
| NumPy            | Numerical computation      |
| Matplotlib       | Data visualization         |
| Seaborn          | EDA and visualization      |
| Scikit-learn     | Preprocessing and ML       |
| XGBoost          | Classification             |
| LightGBM         | Final classification model |
| CatBoost         | Classification             |
| Joblib           | Model serialization        |
| Streamlit        | Frontend                   |
| Jupyter Notebook | Model development          |
| Git & GitHub     | Version control            |

---

# 📦 Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
```

Move into the project directory:

```bash
cd Employee-Salary-Prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

# 📋 Requirements

Your `requirements.txt` can contain:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
catboost
joblib
streamlit
```

---

# 🔮 Future Improvements

* Deploy the Streamlit application online
* Improve model performance with additional feature engineering
* Add probability/confidence of prediction
* Add interactive EDA visualizations to the frontend
* Improve the UI/UX of the Streamlit application
* Add an API using FastAPI
* Add model monitoring after deployment

---

# 👨‍💻 Author

**Arjun Singh Tomar**

B.Tech Computer Science & Engineering

---

# ⭐ Conclusion

This project demonstrates a complete Machine Learning classification workflow for predicting whether an individual's annual income is **greater than $50K or less than or equal to $50K**.

The project covers:

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Feature Selection
* Label Encoding
* Ordinal Encoding
* One-Hot Encoding
* Feature Scaling
* Multiple Classification Algorithms
* Model Comparison
* Hyperparameter Tuning
* LightGBM
* Model Serialization
* Streamlit Deployment

The final tuned **LightGBM classifier** provides approximately **87.37% accuracy and 0.9268 ROC-AUC** on the test data.
