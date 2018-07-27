import sys
sys.path.append("/home/gunny/arundo/automl-benchmarking/")

import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from src.hypersearch import xgbc_random
from src.hypersearch import xgbr_random
from config.xgboost import config_dict


def test_xgbc_easy():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=0.75, test_size=0.25)
    best_kwargs, best_score = xgbc_random(config_dict, 5, X_train, y_train, X_test, y_test)
    print(best_kwargs)
    print(best_score)

def test_xgbr_easy():
    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, train_size=0.75, test_size=0.25)
    best_kwargs, best_error = xgbr_random(config_dict, 5, X_train, y_train, X_test, y_test)
    print(best_kwargs)
    print(best_error)

def test_xgbr_easy_2():
    boston = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, train_size=0.75, test_size=0.25)
    best_kwargs, best_error = xgbr_random(config_dict, 5, X_train, y_train, X_test, y_test)
    print(best_kwargs)
    print(best_error)
