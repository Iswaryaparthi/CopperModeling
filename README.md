# CapstoneProject5
### Industrial Copper Modeling

#### Introduction:
The Problem Statement about copper industry deals with  complex data related to sales and pricing. However, this data  suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data. 

Here we need to build two Machine Learning models to predict Selling price and Status. Selling Price is a continuous variable, so used Regression model for prediction. Whereas, Status, is a categorical variable, so used Classification Model.

This dataset contains more noise and linearity between independent variables so it will perform well only with tree based models.

So,used Decision tree Regressor and Decision Tree Classifier algorithms for model building.
Finally, either of the predicted model should been display in Streamlit Web Application for the user to manipualte with the independent variables and get the results either Selling Price or the Status as Won or Lost.

#### Libraries Used:

* import pandas as pd 
* import numpy as np 
* import matplotlib.pyplot as plt
* import seaborn as sns
* from sklearn.preprocessing import OneHotEncoder
* from sklearn.compose import ColumnTransformer
* from sklearn.compose import make_column_selector as selector
* from sklearn.tree import DecisionTreeRegressor
* from sklearn.tree import DecisionTreeClassifier
* from sklearn.model_selection import train_test_split, GridSearchCV
* from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
* from sklearn.metrics import mean_squared_error, r2_score
* import pickle
* import streamlit as st

##### Data Preprocessing:

- Handle Null values
- Clean the noisy data mainly Material reference column
- Mixed datatypes conversion
- Handle zero , non positive values

#### EDA:
- Checking for Outliers through Boxplot, Histplot
- Checking for Skewness in the dataset
- Transform the high positive skewness through log transformation approach
- Before and After Skewness through visualizations
- Correlation among the variables through Heat Map
- Encoding the necessary independent variables
- Used One hot encoder for encoding

##### Feature Engineering:
- HyperParameter tuning
- Grid SearchCV was used to get the best parameters

##### Model building and Evaluation

Selling Price Prediction:

* USed Decision Tree Regressor
* 

##### Model GUI
