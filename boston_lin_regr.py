import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from pandas import DataFrame, Series
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

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
dataset_X.drop(['B'], axis=1, inplace=True)
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
# plt.pause(0)

"""
sns.pairplot(dataset_X[['CRIM','ZN','INDUS']], diag_kind='kde')
sns.pairplot(dataset_X, diag_kind='kde')
plt.show()
plt.pause(0)
input('Press [Enter]')
"""
'''
Here for reference the meaning of the variables, taken from the dataset description https://bit.ly/3yqTZnL

    CRIM - per capita crime rate by town
    ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
    INDUS - proportion of non-retail business acres per town.
    CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    NOX - nitric oxides concentration (parts per 10 million)
    RM - average number of rooms per dwelling
    AGE - proportion of owner-occupied units built prior to 1940
    DIS - weighted distances to five Boston employment centres
    RAD - index of accessibility to radial highways
    TAX - full-value property-tax rate per $10,000
    PTRATIO - pupil-teacher ratio by town
    B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    LSTAT - % lower status of the population
    MEDV - Median value of owner-occupied homes in $1000's
    
We start with a simple (that is, uni-variate) linear regression of MEDV against LSTAT. 
'''

'''
Distribution of LSTAT values
'''

plt.hist(dataset['LSTAT'], bins=50)
# plt.pause(0)

# Performa the simple linear regression of MEDV against LSTAT
X, y = dataset['LSTAT'].values.reshape(-1, 1), dataset['MEDV'].values
model: LinearRegression = LinearRegression().fit(X, y)
y_pred = model.predict(X=X)
slope = model.coef_[0]
intercept = model.intercept_
print(f'\nMEDV linear regression against LSTAT has coefficient (slope) {slope} and intercept {intercept}')
r2 = model.score(X, y)
rss = np.sum(np.power(y - y_pred, 2))
rse = (rss / (n_samples - 2)) ** .5
print(f'RSS={rss:.3f}; RSE={rse:.3f} on {n_samples - 2} degrees of freedom; R-squared={r2:.3f}')

_, ax = plt.subplots()
ax.scatter(X, y, facecolor='none', edgecolor='blue')
ax.plot(X, y_pred, color='black')
ax.set_xlabel('LSTAT')
ax.set_ylabel('MEDV')
ax.set_title('Linear Regression of MEDV over LSTAT')
# plt.pause(0)

''' 
IS there a relationship between MEDV and LSTAT? The linear regression plot indicates so, in spite of outliers toward 
high MEDV values; however, more formally, we want to test the null hypothesis
    -There is no relationship between MEDV and LSTAT
'''

''' We can get a feeling of how accurate the estimate of slope and intercept are, by determining a confidence interval 
for them '''
X_mean = np.mean(X)
se2_slope = rse ** 2 / np.sum(np.power(X - X_mean, 2))
se2_intercept = rse ** 2 * (1 / n_samples + X_mean ** 2 / np.sum(np.power(X - X_mean, 2)))
print(
    f'The 95% confidence interval for the slope is [{slope - 2 * se2_slope ** .5:.3f}, {slope + 2 * se2_slope ** .5:.3f}]')
print(
    f'The 95% confidence interval for the intercept is [{intercept - 2 * se2_intercept ** .5:.3f}, {intercept + 2 * se2_intercept ** .5:.3f}]')

x = np.linspace(np.min(X), np.max(X), n_samples)
y_est = slope * x + intercept
# y_err = x.std() * np.sqrt(1 / len(x) + (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
y_err = rse * np.sqrt(1 / n_samples + (X - X_mean) ** 2 / np.sum((X - X_mean) ** 2))
y_err = y_err.squeeze()

_, ax = plt.subplots()
ax.plot(x, y_est, '-')
ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
ax.plot(X, y, 'o', color='tab:brown')
# plt.pause(0)

# Let's do it again, this time with Statsmodel

X_exog = sm.add_constant(X, prepend=False)  # TODO this is needed for correct result, but why?
model2 = sm.OLS(y, X_exog)
res = model2.fit()
print('\n', res.summary())

residuals = y_pred - y
_, ax = plt.subplots()
ax.set_title('Empirical distribution of residuals')
ax.set_ylabel('Binned distr.')
ax.hist(residuals, 50)
sorted_residuals = np.sort(residuals)
cum_prob = 1. * np.arange(len(sorted_residuals)) / float(len(sorted_residuals) - 1)
# prob_dens = cum_prob /
ax2 = ax.twinx()
ax2.set_ylabel('Cumulative distr.', color='red')
ax2.plot(sorted_residuals, cum_prob, color='red')
# plt.pause(0)
'''
TODO
- My confidence intervale is 95% instead of 97.5%, difficult to compare with those provided by R and Statsmodel;
also my CI seem inconsistent with those provided by R and Statsmodel
- Not clear how to compute and plot the bands of uncertainty around the linear regression, i.e. the CI for a given
inference; see y_err above
Note: se2_intercept and se2_slope appear to be correct when compared to R output
'''

# Now let's do some multi-variate regression

X, y = dataset_X, dataset_y
mv_model: LinearRegression = LinearRegression().fit(X, y)
coeffs = pd.Series(mv_model.coef_, index=mv_model.feature_names_in_)
print(coeffs.to_frame().T)

y_pred = mv_model.predict(X)
mse = mean_squared_error(y, y_pred)

r2 = mv_model.score(X, y)
rss = np.sum(np.power(y - y_pred, 2))
rse = (rss / (n_samples - 2)) ** .5
print(f'RSS={rss:.3f}; RSE={rse:.3f} on {n_samples - 2} degrees of freedom; R-squared={r2:.3f}')
corr_matrix = dataset.corr()
print(corr_matrix)

# Display the correlation matrix as a heatmap with Seaborn
plt.subplots()
corr_matrix_rounded = corr_matrix.round(decimals=1)
ax = sns.heatmap(corr_matrix_rounded,
                 vmin=-1,
                 vmax=1,
                 center=0,
                 cmap=sns.diverging_palette(20, 220, n=200),
                 square=True,
                 annot=True)
ax.set_xticklabels(ax.get_xticklabels(),
                   rotation=45,
                   horizontalalignment='right')
# fig.tight_layout()
plt.pause(0)
