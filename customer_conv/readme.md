# Predicting Customer Conversion for Insurance

## Project Overview

This project aims to predict customer conversion for an insurance company, specifically identifying which customers are likely to purchase insurance and which are not. The process includes cleaning and preprocessing data, exploratory data analysis, handling class imbalance, feature scaling, and building a predictive model using Naive Bayes.

---

## Step-by-Step Explanation of the Code

### Step 1: Importing Necessary Libraries

The project begins by importing essential Python libraries like pandas for data manipulation, seaborn and matplotlib for visualization, scikit-learn for preprocessing and model building, and imbalanced-learn for addressing class imbalance. These libraries provide the tools needed for data analysis, visualization, and machine learning.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
```

---

### Step 2: Loading the Dataset

The dataset is loaded using the `pd.read_csv()` function. The file path points to a local CSV file containing customer data. After loading, the first few rows are displayed using `df.head()` to understand the structure of the dataset.

```python
file = '/path/to/customers_dataset.csv'
df = pd.read_csv(file)
df.head()
```

---

### Step 3: Checking for Missing Values

To ensure data quality, the code checks for missing values in each column using `df.isnull().sum()`. Handling missing values is critical to avoid errors during analysis and modeling.

```python
sumofnull = df.isnull().sum()
```

---

### Step 4: Understanding the Dataset

Data types, statistical summaries, and dataset shape are examined to understand the dataset's structure and characteristics.

```python
print("Dataset datatypes:", df.dtypes)
print("Data description", df.describe())
print("Data shape", df.shape)
```

- **Purpose**: These steps provide insights into the data, including numerical distributions, ranges, and data types.

---

### Step 5: Analyzing Target Distribution

The target variable (`y`) distribution is examined using `value_counts()` to identify class imbalance. This step ensures appropriate handling of imbalanced datasets.

```python
distribute = df['y'].value_counts()
```

---

### Step 6: Handling Duplicates

Duplicate rows are identified using `df.duplicated()` and removed with `df.drop_duplicates()`. This step ensures data integrity and avoids bias from repeated data points.

```python
duplicates = df[df.duplicated()]
df = df.drop_duplicates()
```

---

### Step 7: Exploratory Data Analysis (EDA)

Several visualizations are created to understand the distribution of key features:

1. **Target Variable Distribution**:

   - A count plot of `y` shows the distribution of converted and non-converted customers.

2. **Job Distribution**:

   - The distribution of customers by job type is visualized.

3. **Marital Status Distribution**:

   - A count plot of marital status provides insights into the marital demographics.

4. **Education Level Distribution**:

   - The distribution of education levels among customers is analyzed.

5. **Call Type Distribution**:

   - Call types are visualized to understand their usage patterns.

6. **Month Distribution**:

   - The distribution of campaign months is examined.

7. **Previous Outcome Distribution**:

   - The outcome of previous campaigns is visualized.

```python
sns.countplot(x="y", data=df, palette="pastel")
sns.countplot(x="job", data=df, palette="viridis")
sns.countplot(x="marital", data=df, palette="pastel")
sns.countplot(x="education_qual", data=df, palette="viridis")
sns.countplot(x="call_type", data=df, palette="pastel")
sns.countplot(x="mon", data=df, palette="viridis")
sns.countplot(x="prev_outcome", data=df, palette="pastel")
```

---

### Step 8: Encoding Categorical Variables

Categorical columns are encoded into numerical values using `LabelEncoder`. This step is essential for machine learning algorithms that require numerical inputs.

```python
le = LabelEncoder()
categorical_columns = ['job', 'marital', 'education_qual', 'call_type', 'mon', 'prev_outcome', 'y']
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
```

---

### Step 9: Addressing Class Imbalance

Class imbalance in the target variable (`y`) is addressed using SMOTE (Synthetic Minority Oversampling Technique). SMOTE generates synthetic samples to balance the dataset.

```python
smote = SMOTE(random_state=42)
X = df.drop('y', axis=1)
y = df['y']
X_res, y_res = smote.fit_resample(X, y)
```

---

### Step 10: Standardizing Features

Numerical features are standardized using `StandardScaler` to ensure all features contribute equally to the model.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)
```

---

### Step 11: Splitting the Dataset

The dataset is split into training and testing sets using an 80-20 ratio. The `random_state` ensures reproducibility.

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_res, test_size=0.2, random_state=42)
```

---

### Step 12: Building and Evaluating the Model

A Naive Bayes model is trained and evaluated using metrics like accuracy, precision, recall, and F1-score.

```python
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy * 100)
print("Precision:", precision * 100)
print("Recall:", recall * 100)
print("F1 Score:", f1 * 100)
```

---

## Observations and Possible Reasons for Each Step

1. **Handling Missing Values**: Ensures data quality.
2. **EDA**: Provides insights into patterns and informs feature engineering.
3. **Encoding**: Prepares data for machine learning algorithms.
4. **SMOTE**: Balances classes for fair training.
5. **Scaling**: Prevents dominance of features with larger ranges.
6. **Model Selection**: Gaussian Naive Bayes is simple, interpretable, and works well with categorical features.

---

## Key Results

- Metrics Obtained:

  - Accuracy: \~74%
  - Precision: \~67%
  - Recall: \~91%
  - F1 Score: \~77%

- **Comparison with Expected Results**:

  - Variations might be due to differences in preprocessing, train-test splits, or SMOTE sampling.

---

## Next Steps

1. Experiment with different models (e.g., Random Forest, Logistic Regression).
2. Perform hyperparameter tuning for improved performance.
3. Use feature selection techniques to identify the most critical variables.
4. Validate the model on an unseen test set to ensure generalizability.

I experimented with different models LDA, Decision Trees, Extra Trees, Random Forest with Random Forest scoring about 95% in all metrics tests followed by Extra Trees with the GaussianNB in last place. This tells us that the RF model is the best for this evaluation.

- Random Forest Classifier

  - Accuracy: \~93.3%
  - Precision: \~91%
  - Recall: \~95%
  - F1 Score: \~92%
