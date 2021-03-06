from sklearn.linear_model import LinearRegression
import utils
from numpy import argmax


def test_load_boston_housing_market():
    dataset_X, dataset_y, dataset = utils.load_boston_housing_market()
    assert dataset_X.shape == (506, 12)
    assert dataset_y.shape == (506,)
    assert dataset.shape == (506, 13)
    assert list(dataset.columns) == ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                                     'PTRATIO', 'LSTAT', 'MEDV']


def test_forward_select():
    X, y, _ = utils.load_boston_housing_market()
    selected, r2 = utils.forward_select(LinearRegression, X, y)
    assert selected == ['LSTAT', 'RM', 'PTRATIO', 'DIS', 'NOX', 'CHAS', 'ZN', 'CRIM', 'RAD', 'TAX', 'AGE', 'INDUS']
    i_selected = argmax(r2)
    assert i_selected == 9
    assert r2[i_selected] == 0.7288734084410414

