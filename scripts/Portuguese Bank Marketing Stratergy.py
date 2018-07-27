
# coding: utf-8

# # Portuguese Bank Marketing Stratergy- TPOT Tutorial

# The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.
# https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

# In[1]:


# Import required libraries
from tpot import TPOTClassifier
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np


# In[2]:


#Load the data
Marketing=pd.read_csv('data/Data_FinalProject.csv')
# Marketing.head(5)


# # # Data Exploration
#
# # In[3]:
#
#
# Marketing.groupby('loan').y.value_counts()
#
#
# # In[4]:
#
#
# Marketing.groupby(['loan','marital']).y.value_counts()
#

# # Data Munging

# The first and most important step in using TPOT on any data set is to rename the target class/response variable to class.

# In[5]:


Marketing.rename(columns={'y': 'class'}, inplace=True)


# At present, TPOT requires all the data to be in numerical format. As we can see below, our data set has 11 categorical variables
# which contain non-numerical values: job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome, class.

# In[6]:


# Marketing.dtypes


# We then check the number of levels that each of the five categorical variables have.

# In[7]:


# for cat in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome' ,'class']:
#     print("Number of levels in category '{0}': \b {1:2.2f} ".format(cat, Marketing[cat].unique().size))


# As we can see, contact and poutcome have few levels. Let's find out what they are.

# In[8]:


# for cat in ['contact', 'poutcome','class', 'marital', 'default', 'housing', 'loan']:
#     print("Levels for catgeory '{0}': {1}".format(cat, Marketing[cat].unique()))


# We then code these levels manually into numerical values. For nan i.e. the missing values, we simply replace them with a placeholder value (-999). In fact, we perform this replacement for the entire data set.

# In[9]:


Marketing['marital'] = Marketing['marital'].map({'married':0,'single':1,'divorced':2,'unknown':3})
Marketing['default'] = Marketing['default'].map({'no':0,'yes':1,'unknown':2})
Marketing['housing'] = Marketing['housing'].map({'no':0,'yes':1,'unknown':2})
Marketing['loan'] = Marketing['loan'].map({'no':0,'yes':1,'unknown':2})
Marketing['contact'] = Marketing['contact'].map({'telephone':0,'cellular':1})
Marketing['poutcome'] = Marketing['poutcome'].map({'nonexistent':0,'failure':1,'success':2})
Marketing['class'] = Marketing['class'].map({'no':0,'yes':1})


# In[10]:


Marketing = Marketing.fillna(-999)
pd.isnull(Marketing).any()


# For other categorical variables, we encode the levels as digits using Scikit-learn's MultiLabelBinarizer and treat them as new features.

# In[11]:


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

job_Trans = mlb.fit_transform([{str(val)} for val in Marketing['job'].values])
education_Trans = mlb.fit_transform([{str(val)} for val in Marketing['education'].values])
month_Trans = mlb.fit_transform([{str(val)} for val in Marketing['month'].values])
day_of_week_Trans = mlb.fit_transform([{str(val)} for val in Marketing['day_of_week'].values])


# In[12]:


# day_of_week_Trans


# Drop the unused features from the dataset.

# In[13]:


marketing_new = Marketing.drop(['marital','default','housing','loan','contact','poutcome','class','job','education','month','day_of_week'], axis=1)


# In[14]:


assert (len(Marketing['day_of_week'].unique()) == len(mlb.classes_)), "Not Equal" #check correct encoding done


# In[15]:


# Marketing['day_of_week'].unique(),mlb.classes_


# We then add the encoded features to form the final dataset to be used with TPOT.

# In[16]:


marketing_new = np.hstack((marketing_new.values, job_Trans, education_Trans, month_Trans, day_of_week_Trans))


# In[17]:


# np.isnan(marketing_new).any()


# Keeping in mind that the final dataset is in the form of a numpy array, we can check the number of features in the final dataset as follows.

# In[18]:


# marketing_new[0].size


# Finally we store the class labels, which we need to predict, in a separate variable.

# In[20]:


marketing_class = Marketing['class'].values


# # Data Analysis using TPOT

# To begin our analysis, we need to divide our training data into training and validation sets. The validation set is just to give us an idea of the test set error. The model selection and tuning is entirely taken care of by TPOT, so if we want to, we can skip creating this validation set.

# In[21]:


training_indices, validation_indices = training_indices, testing_indices = train_test_split(Marketing.index, stratify = marketing_class, train_size=0.75, test_size=0.25)
# training_indices.size, validation_indices.size


# After that, we proceed to calling the `fit()`, `score()` and `export()` functions on our training dataset.
# An important TPOT parameter to set is the number of generations (via the `generations` kwarg). Since our aim is to just illustrate the use of TPOT, we assume the default setting of 100 generations, whilst bounding the total running time via the `max_time_mins` kwarg (which may, essentially, override the former setting). Further, we enable control for the maximum amount of time allowed for optimization of a single pipeline, via `max_eval_time_mins`.
#
# On a standard laptop with 4GB RAM, each generation takes approximately 5 minutes to run. Thus, for the default value of 100, without the explicit duration bound, the total run time could be roughly around 8 hours.

# In[22]:


from config.classifier_models_only import classifier_config_dict
time_allocated = 60


# In[23]:


tpot = TPOTClassifier(
    verbosity=3,
    max_time_mins=time_allocated,
    config_dict=classifier_config_dict,
    scoring="neg_log_loss",
    n_jobs=8)
tpot.fit(marketing_new[training_indices], marketing_class[training_indices])


# In the above, 4 generations were computed, each giving the training efficiency of fitting model on the training set. As evident, the best pipeline is the one that has the CV score of 91.373%. The model that produces this result is one that fits a decision tree algorithm on the data set. Next, the test error is computed for validation purposes.

# In[24]:


tpot.score(marketing_new[validation_indices], Marketing.loc[validation_indices, 'class'].values)


# In[23]:


tpot.export('tpot_marketing_pipeline.py')


# In[ ]:


# # %load tpot_marketing_pipeline.py
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
#
# # NOTE: Make sure that the class is labeled 'target' in the data file
# tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
# features = tpot_data.drop('target', axis=1).values
# training_features, testing_features, training_target, testing_target =             train_test_split(features, tpot_data['target'].values, random_state=42)
#
# # Score on the training set was:0.913728927925
# exported_pipeline = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=16, min_samples_split=8)
#
# exported_pipeline.fit(training_features, training_target)
# results = exported_pipeline.predict(testing_features)


# ## XGBOOST Experiment

# In[25]:

#
# import os
# os.getcwd()


# In[26]:


import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from src.hypersearch import xgbc_random
from config.xgboost import config_dict


# In[29]:


print(xgbc_random(
    config_dict,
    3600,
    marketing_new[training_indices],
    marketing_class[training_indices],
    marketing_new[validation_indices],
    Marketing.loc[validation_indices, 'class'].values))
