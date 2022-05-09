import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from pandas import DataFrame, Series
from sklearn import datasets
from matplotlib import pyplot as plt

plt.ioff()

"""
This is an exploration of linear regression, and various tools available to solve it, with Python
The dataset of choice is the Boston Housing Prices http://lib.stat.cmu.edu/datasets/boston
"""

# Fetch the dataset from its original source
# run block of code and catch warnings
with warnings.catch_warnings():
    # ignore all caught warnings
    warnings.filterwarnings("ignore")
    boston = datasets.load_boston()

dataset_X = DataFrame(boston.data, columns=boston.feature_names)
dataset_y = Series(boston.target)
dataset = dataset_X.copy()
dataset['MEDV'] = dataset_y
n_samples = len(dataset)
n_vars = len(dataset.columns) - 1
print(f'\nThe dataset contains {n_samples} samples, and {n_vars} variables (target not included).')

pd.set_option('display.max_columns', None)  # Print all the columns in the dataframe
pd.set_option('display.expand_frame_repr', False)  # Prevents line breaks while printing the dataframe

print('\nThe first 5 samples from the dataset')
print(dataset.head())

print('\nDataset stats')
print(dataset.describe().T)

print('\nCount of unique values for each variable')
counts = dataset.nunique()
print(counts.to_frame().T)

print('\nCount number of NaN for each variable')
print(dataset.isna().sum().to_frame().T)

# Draw box plots for every variable, one under another (each has its own scale)
# fig, axes = plt.subplots(len(dataset.columns), 1)
fig, axes = plt.subplots(1, n_vars + 1)
fig.subplots_adjust(wspace=.5)
for i, col in enumerate(dataset.columns):
    bplot = sns.boxplot(y=dataset[col], ax=axes[i])
    bplot.set(ylabel=None)
    bplot.set(xlabel=col)
plt.pause(0)

"""
sns.pairplot(dataset_X[['CRIM','ZN','INDUS']], diag_kind='kde')
sns.pairplot(dataset_X, diag_kind='kde')
plt.show()
plt.pause(0)
input('Press [Enter]')
"""
