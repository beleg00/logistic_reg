# Income Classification Task

This repository contains a solution to the income classification problem. The task is to classify individuals based on their income level using logistic regression and support vector machine (SVM) models. The dataset used for this task is the "Adult" dataset from the UCI Machine Learning Repository.

## Dataset

The dataset contains the following columns:
- `age`: Age of the individual
- `workclass`: Work class of the individual
- `fnlwgt`: Final weight
- `education`: Education level
- `education-num`: Number of years of education
- `marital-status`: Marital status
- `occupation`: Occupation
- `relationship`: Relationship status
- `race`: Race of the individual
- `sex`: Gender of the individual
- `capital-gain`: Capital gain
- `capital-loss`: Capital loss
- `hours-per-week`: Hours worked per week
- `native-country`: Native country
- `income`: Income level (<=50K or >50K)

## Task Description

The task is to predict whether an individual's income is greater than 50K or less than or equal to 50K based on the provided attributes. This is a binary classification problem. 

## Solution

The solution involves the following steps:

1. **Data Loading and Preprocessing**: Load the dataset, handle missing values, and preprocess the data.
2. **Data Visualization**: Visualize the distribution of income with respect to various attributes.
3. **Feature Transformation**: Transform categorical features using OneHotEncoder and scale numerical features using StandardScaler.
4. **Model Training**: Train logistic regression and SVM models using pipelines.
5. **Model Evaluation**: Evaluate the models' accuracy on the test set.

## Code

The code implementation is as follows:

```python
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]
data = pd.read_csv(url, names=column_names, na_values=' ?', skipinitialspace=True)

# Data exploration
print(data.head())
print(data.info())
print(data.describe())
print(data['income'].value_counts())

# Handle missing values
data.dropna(inplace=True)

# Data visualization
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='age', hue='income', multiple='stack')
plt.title('Income distribution by age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='sex', hue='income')
plt.title('Income distribution by gender')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# Feature transformation
X = data.drop('income', axis=1)
y = data['income'].apply(lambda x: 1 if x == '>50K' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Logistic regression pipeline
logreg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression(max_iter=1000))])

# SVM pipeline
svc_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', SVC(kernel='linear'))])

# Train and evaluate logistic regression
logreg_pipeline.fit(X_train, y_train)
logreg_predictions = logreg_pipeline.predict(X_test)
logreg_accuracy = accuracy_score(y_test, logreg_predictions)
print(f'Logistic regression accuracy: {logreg_accuracy:.4f}')

# Train and evaluate SVM
svc_pipeline.fit(X_train, y_train)
svc_predictions = svc_pipeline.predict(X_test)
svc_accuracy = accuracy_score(y_test, svc_predictions)
print(f'Accuracy of the support vector method: {svc_accuracy:.4f}')
```

## Requirements

- pandas
- matplotlib
- seaborn
- scikit-learn

You can install the required packages using the following command:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

## Usage

To run the code, simply execute the script in your Python environment. Ensure that you have an active internet connection to download the dataset.

## Results

The accuracy of the logistic regression and SVM models will be printed in the console output.
