import warnings
from sklearn import datasets
from pandas import DataFrame, Series
from copy import deepcopy


def load_boston_housing_market():
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
        boston = datasets.load_boston()

    """
    dataset_X => predictors
    dataset_y => target variable
    dataset => predictors and target variable
    """
    dataset_X = DataFrame(boston.data, columns=boston.feature_names)
    """
    Dropping the 'B' variable from the dataset: it is a non-invertible variable engineered by the authors of the dataset; it is not clear how to use it for the analysis to follow, whatever the usage intended by the dataset authors.
    """
    dataset_X.drop('B', axis=1, inplace=True)
    dataset_y = Series(boston.target)
    dataset = dataset_X.copy()
    dataset['MEDV'] = dataset_y  # Assign the correct name to the target variable
    return dataset_X, dataset_y, dataset


def forward_select(model, X, y):
    selected = []
    candidates = list(X.columns)  # Columns not yet selected
    best_adj_r2 = -1
    all_adj_r2 = []
    while candidates:
        best_candidate = None
        # Find among the variables not yet selected for the model which one increases its R-squared the most
        best_r2 = -1  # When leaving the loop, will contain the R-squared for the model with the best variable selected
        adj_r2_best_r2 = -1  # Will contain the adjusted R-squared for the model with the best variable selected
        for candidate in candidates:
            X_selected = X[selected + [candidate]].copy()
            res = model().fit(X_selected, y)
            r_2 = res.score(X_selected, y)
            if r_2 > best_r2:
                best_candidate = candidate
                best_r2 = r_2
                n_vars = len(X_selected.columns)
                n = len(X_selected)
                adj_r2_best_r2 = 1 - (1 - r_2) * (n - 1) / (n - n_vars - 1)
        assert best_candidate is not None
        selected.append(best_candidate)
        candidates = [item for item in candidates if item != best_candidate]
        all_adj_r2.append(adj_r2_best_r2)
    return selected, all_adj_r2


if __name__ == '__main__':
    ...
