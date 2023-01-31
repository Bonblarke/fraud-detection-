#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from sklearn.preprocessing import StandardScaler


# In[ ]:


# Load the transaction data into a Pandas dataframe
df = pd.read_csv('transactions.csv')


# In[ ]:


# Pre-process the data
df = df.dropna()
df = df[df['amount'] > 0]


# In[ ]:


# Create new features
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_month'] = df['date'].dt.day
df['week_of_year'] = df['date'].dt.weekofyear



# In[ ]:


# Split the data into a training set and a testing set
train = df.sample(frac=0.8, random_state=1)
test = df.drop(train.index)



# In[ ]:


# Scale the data
scaler = StandardScaler()
train[['amount', 'day_of_week', 'day_of_month', 'week_of_year']] = scaler.fit_transform(train[['amount', 'day_of_week', 'day_of_month', 'week_of_year']])
test[['amount', 'day_of_week', 'day_of_month', 'week_of_year']] = scaler.transform(test[['amount', 'day_of_week', 'day_of_month', 'week_of_year']])


# In[ ]:


# Train the KNN anomaly detection model
clf = KNN()
clf.fit(train[['amount', 'day_of_week', 'day_of_month', 'week_of_year']])


# In[ ]:


# Predict the class labels (fraud or not fraud) for the test data
y_test = clf.predict(test[['amount', 'day_of_week', 'day_of_month', 'week_of_year']])


# In[ ]:


# Print the number of fraud cases detected
print('Number of Fraud Cases:', np.sum(y_test))



# This code uses the PyOD library's KNN anomaly detection model to detect fraudulent transactions in a financial services company. The transaction data is pre-processed and split into a training set and a testing set, and new features are created from the existing data. The KNN model is trained on the training data, and the class labels (fraud or not fraud) are predicted for the test data. The number of fraud cases detected is printed.
# 
