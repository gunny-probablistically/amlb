import time
import logging

import numpy as np
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error


logging.basicConfig(
    filename="hypersearch.log",
    level=logging.INFO,
    format="%(asctime)s --- %(levelname)s --- %(message)s"
)

def xgbc_random(kwargs: dict, max_duration: int, train_X, train_y, test_X, test_y):
    """Time-constrained random search for best hyperparams for XGBClassifier

    Args:
        - kwargs: Range of each hyperparam to try for XGBClassifier
        - max_duration: Duration in seconds to attempt search.

    Returns:
        - best_kwargs: Combination of hyperparams with best results
            achieved within search `max_duration`.
    """

    best_kwargs = None
    best_score = None
    used_kwargs = []
    start = time.time()
    while time.time() - start < max_duration:
        logging.info("Trace xgbc_random() --- starting new iteration --- time elapsed = {} seconds".format(time.time() - start))
        try_kwargs = {k:np.random.choice(v) for (k, v) in kwargs.items()}
        if try_kwargs not in used_kwargs:
            logging.info("Trace xgbc_random() --- trying hyperparameters = {}".format(try_kwargs))
            used_kwargs.append(try_kwargs)
            classifier = XGBClassifier(**try_kwargs)
            classifier.fit(train_X, train_y, verbose=False)
            pred_y = classifier.predict_proba(test_X)
            score = log_loss(test_y, pred_y)
            if best_score is None or score < best_score:
                best_score = score
                best_kwargs = try_kwargs
                logging.info("Trace xgbc_random() --- best_score updated to {}".format(best_score))
                logging.info("Trace xgbc_random() --- best_kwargs updated to {}".format(best_kwargs))
        else:
            logging.info("Trace xgbc_random() --- skipping hyperparameters --- they have been tried.")
            continue
    logging.info("Trace xgbc_random() --- duration exceeded --- process quitting with best_score = {}".format(best_score))
    logging.info("Trace xgbc_random() --- duration exceeded --- process quitting with best_kwargs = {}".format(best_kwargs))
    return best_kwargs, best_score

def xgbr_random(kwargs: dict, max_duration: int, train_X, train_y, test_X, test_y):
    """Time constrained random search for best hyperparams for XGBRegressor

    Args:
        - kwargs: Dict<List>. Range of each hyperparam to try for XGBRegressor
        - max_duration: Duration in seconds to attempt search.

    Returns:
        - best_kwargs: Combination of hyperparams with best results
            achieved within search `max_duration`.
    """

    best_kwargs = None
    best_error = None
    used_kwargs = []
    start = time.time()
    while time.time() - start < max_duration:
        logging.info("Trace xgbr_random() --- starting new iteration --- time elapsed = {} seconds".format(time.time() - start))
        try_kwargs = {k:np.random.choice(v) for (k, v) in kwargs.items()}
        if try_kwargs not in used_kwargs:
            logging.info("Trace xgbr_random() --- trying hyperparameters = {}".format(try_kwargs))
            used_kwargs.append(try_kwargs)
            classifier = XGBRegressor(**try_kwargs)
            classifier.fit(train_X, train_y, verbose=False)
            pred_y = classifier.predict(test_X)
            error = mean_squared_error(test_y, pred_y)
            if not best_error or error < best_error:
                best_error = error
                best_kwargs = try_kwargs
                logging.info("Trace xgbr_random() --- best_error updated to {}".format(best_error))
                logging.info("Trace xgbr_random() --- best_kwargs updated to {}".format(best_kwargs))
        else:
            logging.info("Trace xgbr_random() --- skipping hyperparameters --- they have been tried.")
            continue
    logging.info("Trace xgbr_random() --- duration exceeded --- process quitting with best_error = {}".format(best_error))
    logging.info("Trace xgbr_random() --- duration exceeded --- process quitting with best_kwargs = {}".format(best_kwargs))
    return best_kwargs, best_error
