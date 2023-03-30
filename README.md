# MachineLearning
Coverage of supervised and unsupervised learning, neural networks, natural language processing, etc.

## 1. False Discovery Rate (FDR)
False discovery rate (FDR) is a statistical concept that measures the proportion of false positives among all significant test results. In other words, it is the probability that a significant result is actually a false positive.

## 2. Logistic Regression
Logistic regression is a statistical technique used to model the relationship between a categorical dependent variable and one or more independent variables, which can be either categorical or continuous. It is a type of generalized linear model (GLM) that is commonly used for binary classification tasks, where the goal is to predict the probability of an event occurring or not occurring.

## 3. Model Selection
Model selection is the process of choosing the best statistical model from a set of candidate models for a given data set. The goal of model selection is to find a model that can accurately predict the outcome variable or explain the variation in the data while avoiding overfitting or underfitting.

### a. Cross Validation
Cross-validation is a technique used to assess the performance of a statistical model by evaluating its ability to generalize to new data that was not used in the training of the model. It involves dividing the data into training and testing sets and then iteratively training the model on different subsets of the training data and evaluating its performance on the testing data.

### b. Lasso Regression
Lasso regression is a linear regression technique used for feature selection and regularization. It is a type of linear regression that uses an L1 penalty term to shrink the coefficient values towards zero, resulting in a sparse model with only the most important features.

## 4. Regression Discontinuity Design (RDD)
Regression Discontinuity Design (RDD) is a quasi-experimental research design used to estimate the causal effect of a treatment or intervention on an outcome variable. It is a type of design that involves the use of a natural threshold or cutoff point to assign subjects to treatment and control groups, based on their score on a continuous variable.

## 5. k-Nearest Neighbors (kNN)
k-Nearest Neighbors (kNN) is a machine learning algorithm used for classification and regression analysis. It is a non-parametric algorithm that works by comparing a new observation to the k closest known observations in the training dataset and making predictions based on the majority vote (in the case of classification) or the mean (in the case of regression) of their target variables.

## 6. Principal Component Analysis (PCA)
Principal Component Analysis (PCA) is a dimensionality reduction technique used in machine learning and data science to transform high-dimensional datasets into a lower-dimensional space while preserving most of the information.

## 7. Decision Tree
A Decision Tree is a machine learning algorithm used for classification and regression analysis. It is a supervised learning algorithm that works by recursively partitioning the input space into subsets based on the values of the input features, in a way that maximizes the separation between different classes or minimizes the variance of the target variable.

## 8. Random Forest
Random forest is a popular ensemble learning algorithm used in machine learning for classification and regression tasks. It is a combination of multiple decision trees, where each tree is trained on a random subset of the training data and a random subset of the features.

# Final Project - Predictive Modeling to Control Customer Churn Rates
## Overview
In this report, we are going to apply exploratory data analysis to identify patterns and trends that indicate a higher likelihood of churn. It will be used in developing four predictive models.The Random Forest model showed that elderly, couples and male consumers have higher retention rate. Important features identified for marketing campaigns would address specific concerns of at-risk customers. By reducing churn, the company can increase customer satisfaction and loyalty, leading to higher retention rates and increased revenue.

## Data Source
Telco Customer Churn on Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn?datasetId=13996&sortBy=voteCount

## Model Results
| Model Name | Accuracy | Precision |
| ---------- | ---------- | ---------- |
| CATboost | 1.14 |
| XGBOOST | 1.61 | 
| Linear Regression | 6.31 |
| PCA transform | 13.76 | 

