# -*- coding: utf-8 -*-
"""

@author: beleg00

"""

# import lib

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

# Загрузка данных
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]
data = pd.read_csv(url, names=column_names, na_values=' ?', skipinitialspace=True)

data.head()

data.info()
data.describe()
data['income'].value_counts()

# Skip check
data.isnull().sum()

# Deleting skipped lines
data.dropna(inplace=True)

# Alternatively, fill in the blanks with the most frequent values
# data.fillna(data.mode().iloc[0], inplace=True)

# Visualization
# Income distribution by age
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='age', hue='income', multiple='stack')
plt.title('Income distribution by age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Income distribution by gender
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='sex', hue='income')
plt.title('Income distribution by gender')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# Transformation of categorical attributes

# Identification of the target variable and attributes
X = data.drop('income', axis=1)
y = data['income']

# Converting the target variable to binary form
y = y.apply(lambda x: 1 if x == '>50K' else 0)

# Splitting data into training and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Converting categorical attributes with OneHotEncoder
categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Creating a Pipeline for Logistic Regression
logreg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression(max_iter=1000))])

# Pipeline creation for the support vector method
svc_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', SVC(kernel='linear'))])

# Logistic regression training
logreg_pipeline.fit(X_train, y_train)
logreg_predictions = logreg_pipeline.predict(X_test)
logreg_accuracy = accuracy_score(y_test, logreg_predictions)
print(f'Logistic regression accuracy: {logreg_accuracy:.4f}')

# Training of the support vector method
svc_pipeline.fit(X_train, y_train)
svc_predictions = svc_pipeline.predict(X_test)
svc_accuracy = accuracy_score(y_test, svc_predictions)
print(f'Accuracy of the support vector method: {svc_accuracy:.4f}')
