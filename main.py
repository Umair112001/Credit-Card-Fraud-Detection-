# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Step 2: Load and Explore Data
# Load the dataset (Ensure 'creditcard.csv' is downloaded locally)
data = pd.read_csv("creditcard.csv")

# Explore the data
print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset Info:")
print(data.info())
print("\nDescriptive Statistics:")
print(data.describe())

# Plot 1: Class Distribution Plot (Fixed)
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=data, hue='Class', palette='Set1', legend=False)
plt.title('Class Distribution (0: Non-Fraud, 1: Fraud)')
plt.show()


# Step 3: Preprocess Data
# Normalize the 'Amount' column for better results
scaler = StandardScaler()
data['scaled_amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

# Drop original 'Amount' column as we have scaled it
data = data.drop(['Amount'], axis=1)

# Plot 2: Correlation Heatmap (optional: can focus on top features for clarity)
plt.figure(figsize=(16, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.2)
plt.title('Correlation Heatmap')
plt.show()

# Step 4: Split Data into Training and Test Sets
X = data.drop(['Class'], axis=1)  # Features
y = data['Class']  # Target variable (0: non-fraud, 1: fraud)

# Split data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Train the Model (Random Forest)
# Adding GridSearchCV for Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# Using GridSearchCV to find the best parameters
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Use the best estimator found by GridSearchCV
best_rf = grid_search.best_estimator_

# Fit the model with the best parameters
best_rf.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = best_rf.predict(X_test)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot 3: Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

# Classification report and accuracy score
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 7: Feature Importance Plot
importances = best_rf.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

# Plot 4: Feature Importance
plt.figure(figsize=(10, 6))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.tight_layout()
plt.show()
