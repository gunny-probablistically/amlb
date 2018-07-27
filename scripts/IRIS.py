
# coding: utf-8

# In[1]:


from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# Load the IRIS data set and explore its contents.

# In[2]:


iris = load_iris()
iris.data[0:5], iris.target


# Split the data set in train and test.

# In[3]:


X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    train_size=0.75, test_size=0.25)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[5]:


from config.classifier_models_only import classifier_config_dict


# In[6]:


time_allocated = 60
tpot = TPOTClassifier(
    max_time_mins=time_allocated,
    config_dict=classifier_config_dict,
    verbosity=3,
    scoring="neg_log_loss",
    n_jobs=8)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))


# In[5]:


#tpot.export('tpot_iris_pipeline.py')


# In[ ]:


# %load tpot_iris_pipeline.py
# import numpy as np
#
# from sklearn.kernel_approximation import RBFSampler
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline
# from sklearn.tree import DecisionTreeClassifier
#
# # NOTE: Make sure that the class is labeled 'class' in the data file
# tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
# features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
# training_features, testing_features, training_classes, testing_classes =     train_test_split(features, tpot_data['class'], random_state=42)
#
# exported_pipeline = make_pipeline(
#     RBFSampler(gamma=0.8500000000000001),
#     DecisionTreeClassifier(criterion="entropy", max_depth=3, min_samples_leaf=4, min_samples_split=9)
# )
#
# exported_pipeline.fit(training_features, training_classes)
# results = exported_pipeline.predict(testing_features)


#
# # XGBOOST Experiment

# In[7]:


from sklearn.model_selection import train_test_split

from src.hypersearch import xgbc_random
from config.xgboost import config_dict


# In[8]:


xgbc_random(
    config_dict,
    3600,
    X_train,
    y_train,
    X_test,
    y_test)
