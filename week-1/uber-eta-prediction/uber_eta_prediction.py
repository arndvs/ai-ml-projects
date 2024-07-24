
#!/usr/bin/env python3

import sys\

print(sys.executable)
print(sys.path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Display basic information about the training data
print(train_data.info())
print(train_data.describe())

# Check for missing values
print(train_data.isnull().sum())

# Visualize the distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(train_data['Time_taken(min)'], kde=True)
plt.title('Distribution of Time Taken')
plt.show()

# Explore correlations between numerical features
numerical_features = train_data.select_dtypes(include=[np.number]).columns
correlation_matrix = train_data[numerical_features].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
