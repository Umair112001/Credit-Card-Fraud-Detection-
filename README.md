Credit Card Fraud Detection with Random Forest Classifier
This project demonstrates how to detect credit card fraud using a Random Forest Classifier. The dataset used is the Credit Card Fraud Detection Dataset available on Kaggle, which contains transactions made by European cardholders in September 2013. The model is trained to distinguish between fraudulent and legitimate transactions.

Project Overview
The primary objective of this project is to build a machine learning model using the Random Forest algorithm to classify transactions as either fraudulent or non-fraudulent. Additionally, the model undergoes hyperparameter tuning using GridSearchCV to achieve optimal performance.

Key Features
Data Preprocessing: Standardization of the Amount feature and correlation analysis using heatmaps.
Class Imbalance Handling: The dataset is highly imbalanced, with 0.17% of the transactions being fraudulent.
Random Forest Model: Random Forest Classifier is used for prediction. Hyperparameter tuning is performed using GridSearchCV to optimize the model.
Performance Evaluation: Confusion matrix, classification report, and feature importance are used to evaluate the model's performance.
Dataset
Source: The dataset is available on Kaggle Credit Card Fraud Detection.
Features: The dataset consists of 31 features, including the transaction Amount, and the Class label where 0 represents non-fraudulent transactions and 1 represents fraudulent transactions.
Dependencies
Python 3.x
Libraries:
pandas
numpy
matplotlib
seaborn
scikit-learn
