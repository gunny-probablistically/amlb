
# coding: utf-8

# In[1]:


from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# In[2]:


digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[3]:


from config.classifier_models_only import classifier_config_dict
time_allocated = 60


# In[4]:


tpot = TPOTClassifier(
    max_time_mins=time_allocated,
    config_dict=classifier_config_dict,
    verbosity=3,
    scoring="neg_log_loss",
    n_jobs=8)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))


# In[4]:


# tpot.export('tpot_mnist_pipeline.py')


# In[ ]:


# %load tpot_mnist_pipeline.py
# import numpy as np
#
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
#
# # NOTE: Make sure that the class is labeled 'class' in the data file
# tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
# features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
# training_features, testing_features, training_classes, testing_classes =     train_test_split(features, tpot_data['class'], random_state=42)
#
# exported_pipeline = KNeighborsClassifier(n_neighbors=4, p=2, weights="distance")
#
# exported_pipeline.fit(training_features, training_classes)
# results = exported_pipeline.predict(testing_features)


# # XGBOOST Experiment

# In[5]:


import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from src.hypersearch import xgbc_random
from config.xgboost import config_dict


# In[6]:


print(xgbc_random(
    config_dict,
    3600,
    X_train,
    y_train,
    X_test,
    y_test))
