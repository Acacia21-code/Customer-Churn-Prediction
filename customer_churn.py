# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')
# Load dataset
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"Dataset shape: {data.shape}")
data.head()
# Check for missing values
print(data.isnull().sum())

# TotalCharges has some missing values encoded as spaces, convert to numeric
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
print(data['TotalCharges'].isnull().sum())

# Drop rows with missing TotalCharges
data = data.dropna(subset=['TotalCharges'])

# Drop customerID as it is an identifier
data = data.drop('customerID', axis=1)

# Convert target variable 'Churn' to binary
data['Churn'] = data['Churn'].map({'Yes':1, 'No':0})

# Check data types
print(data.dtypes)
# Identify categorical columns
cat_cols = data.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {cat_cols}")

# Binary columns with Yes/No values to 1/0
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

for col in binary_cols:
    data[col] = data[col].map({'Yes':1, 'No':0})

# Encode remaining categorical columns with one-hot encoding
data = pd.get_dummies(data, columns=[col for col in cat_cols if col not in binary_cols], drop_first=True)

print(f"Data shape after encoding: {data.shape}")
# Scale numerical features
scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

data[num_cols] = scaler.fit_transform(data[num_cols])
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check class distribution
print("Before SMOTE:")
print(y_train.value_counts())

# Use SMOTE to balance classes in training set
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("After SMOTE:")
print(pd.Series(y_train_res).value_counts())
# Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)
# Predict on test set
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC AUC score and curve
auc = roc_auc_score(y_test, y_proba)
print(f"ROC AUC Score: {auc:.3f}")

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
# Plot feature importances
importances = rf.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,6))
sns.barplot(x=importances[indices][:15], y=features[indices][:15])
plt.title('Top 15 Feature Importances')
plt.show()
