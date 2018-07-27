import numpy as np

config_dict = {
    'n_estimators': [50, 100, 500, 1000],
    'max_depth': range(1, 11),
    'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
    'subsample': np.arange(0.05, 1.01, 0.05),
    'min_child_weight': range(1, 21),
    'nthread': [8],
    'gamma': np.arange(0, 5.1, 0.5),
    'colsample_bytree': np.arange(0.5, 1.01, 0.1)
}
