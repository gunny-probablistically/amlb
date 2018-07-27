
# coding: utf-8

# # TPOT tutorial on the Titanic dataset

# The Titanic machine learning competition on [Kaggle](https://www.kaggle.com/c/titanic) is one of the most popular beginner's competitions on the platform. We will use that competition here to demonstrate the implementation of TPOT.

# In[3]:


# Import required libraries
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# In[4]:


# Load the data
titanic = pd.read_csv('data/titanic_train.csv')
titanic.head(5)


# ## Data Exploration

# In[3]:


# titanic.groupby('Sex').Survived.value_counts()
#
#
# # In[4]:
#
#
# titanic.groupby(['Pclass','Sex']).Survived.value_counts()


# In[5]:


# id = pd.crosstab([titanic.Pclass, titanic.Sex], titanic.Survived.astype(float))
# id.div(id.sum(1).astype(float), 0)


# ## Data Munging

# The first and most important step in using TPOT on any data set is to rename the target class/response variable to `class`.

# In[6]:


titanic.rename(columns={'Survived': 'class'}, inplace=True)


# At present, TPOT requires all the data to be in numerical format. As we can see below, our data set has 5 categorical variables which contain non-numerical values: `Name`, `Sex`, `Ticket`, `Cabin` and `Embarked`.

# In[7]:


# titanic.dtypes


# We then check the number of levels that each of the five categorical variables have.

# In[8]:


# for cat in ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']:
#     print("Number of levels in category '{0}': \b {1:2.2f} ".format(cat, titanic[cat].unique().size))


# As we can see, `Sex` and `Embarked` have few levels. Let's find out what they are.

# In[9]:


# for cat in ['Sex', 'Embarked']:
#     print("Levels for catgeory '{0}': {1}".format(cat, titanic[cat].unique()))


# We then code these levels manually into numerical values. For `nan` i.e. the missing values, we simply replace them with a placeholder value (-999). In fact, we perform this replacement for the entire data set.

# In[10]:


titanic['Sex'] = titanic['Sex'].map({'male':0,'female':1})
titanic['Embarked'] = titanic['Embarked'].map({'S':0,'C':1,'Q':2})


# In[11]:


titanic = titanic.fillna(-999)
# pd.isnull(titanic).any()


# Since `Name` and `Ticket` have so many levels, we drop them from our analysis for the sake of simplicity. For `Cabin`, we encode the levels as digits using Scikit-learn's [`MultiLabelBinarizer`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html) and treat them as new features.

# In[12]:


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
CabinTrans = mlb.fit_transform([{str(val)} for val in titanic['Cabin'].values])


# In[13]:


# CabinTrans


# Drop the unused features from the dataset.

# In[14]:


titanic_new = titanic.drop(['Name','Ticket','Cabin','class'], axis=1)


# In[15]:


assert (len(titanic['Cabin'].unique()) == len(mlb.classes_)), "Not Equal" #check correct encoding done


# We then add the encoded features to form the final dataset to be used with TPOT.

# In[16]:


titanic_new = np.hstack((titanic_new.values,CabinTrans))


# In[17]:


# np.isnan(titanic_new).any()


# Keeping in mind that the final dataset is in the form of a numpy array, we can check the number of features in the final dataset as follows.

# In[18]:


# titanic_new[0].size


# Finally we store the class labels, which we need to predict, in a separate variable.

# In[19]:


titanic_class = titanic['class'].values


# ## Data Analysis using TPOT

# To begin our analysis, we need to divide our training data into training and validation sets. The validation set is just to give us an idea of the test set error. The model selection and tuning is entirely taken care of by TPOT, so if we want to, we can skip creating this validation set.

# In[20]:


training_indices, validation_indices = training_indices, testing_indices = train_test_split(titanic.index, stratify = titanic_class, train_size=0.75, test_size=0.25)
# training_indices.size, validation_indices.size


# After that, we proceed to calling the `fit`, `score` and `export` functions on our training dataset. To get a better idea of how these functions work, refer the TPOT documentation [here](http://epistasislab.github.io/tpot/api/).
#
# An important TPOT parameter to set is the number of generations. Since our aim is to just illustrate the use of TPOT, we have set it to 5. On a standard laptop with 4GB RAM, it roughly takes 5 minutes per generation to run. For each added generation, it should take 5 mins more. Thus, for the default value of 100, total run time could be roughly around 8 hours.

# In[21]:


from config.classifier_models_only import classifier_config_dict
time_allocated = 60


# In[22]:


tpot = TPOTClassifier(
    max_time_mins=time_allocated,
    config_dict=classifier_config_dict,
    verbosity=3,
    n_jobs=8)
tpot.fit(titanic_new[training_indices], titanic_class[training_indices])


# In[22]:


tpot.score(titanic_new[validation_indices], titanic.loc[validation_indices, 'class'].values)


# In[23]:


tpot.export('tpot_titanic_pipeline.py')


# Let's have a look at the generated code. As we can see, the random forest classifier performed the best on the given dataset out of all the other models that TPOT currently evaluates on. If we ran TPOT for more generations, then the score should improve further.

# In[ ]:


# %load tpot_titanic_pipeline.py
# import numpy as np
#
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
#
# # NOTE: Make sure that the class is labeled 'class' in the data file
# tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
# features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
# training_features, testing_features, training_classes, testing_classes =     train_test_split(features, tpot_data['class'], random_state=42)
#
# exported_pipeline = RandomForestClassifier(bootstrap=False, max_features=0.4, min_samples_leaf=1, min_samples_split=9)
#
# exported_pipeline.fit(training_features, training_classes)
# results = exported_pipeline.predict(testing_features)


# # ### Make predictions on the submission data
#
# # In[25]:
#
#
# # Read in the submission dataset
# titanic_sub = pd.read_csv('data/titanic_test.csv')
# titanic_sub.describe()
#
#
# # The most important step here is to check for new levels in the categorical variables of the submission dataset that are absent in the training set. We identify them and set them to our placeholder value of '-999', i.e., we treat them as missing values. This ensures training consistency, as otherwise the model does not know what to do with the new levels in  the submission dataset.
#
# # In[26]:
#
#
# for var in ['Cabin']: #,'Name','Ticket']:
#     new = list(set(titanic_sub[var]) - set(titanic[var]))
#     titanic_sub.ix[titanic_sub[var].isin(new), var] = -999
#
#
# # We then carry out the data munging steps as done earlier for the training dataset.
#
# # In[27]:
#
#
# titanic_sub['Sex'] = titanic_sub['Sex'].map({'male':0,'female':1})
# titanic_sub['Embarked'] = titanic_sub['Embarked'].map({'S':0,'C':1,'Q':2})
#
#
# # In[28]:
#
#
# titanic_sub = titanic_sub.fillna(-999)
# pd.isnull(titanic_sub).any()
#
#
# # While calling `MultiLabelBinarizer` for the submission data set, we first fit on the training set again to learn the levels and then transform the submission dataset values. This further ensures that only those levels that were present in the training dataset are transformed. If new levels are still found in the submission dataset then it will return an error and we need to go back and check our earlier step of replacing new levels with the placeholder value.
#
# # In[29]:
#
#
# from sklearn.preprocessing import MultiLabelBinarizer
# mlb = MultiLabelBinarizer()
# SubCabinTrans = mlb.fit([{str(val)} for val in titanic['Cabin'].values]).transform([{str(val)} for val in titanic_sub['Cabin'].values])
# titanic_sub = titanic_sub.drop(['Name','Ticket','Cabin'], axis=1)
#
#
# # In[30]:
#
#
# # Form the new submission data set
# titanic_sub_new = np.hstack((titanic_sub.values,SubCabinTrans))
#
#
# # In[31]:
#
#
# np.any(np.isnan(titanic_sub_new))
#
#
# # In[32]:
#
#
# # Ensure equal number of features in both the final training and submission dataset
# assert (titanic_new.shape[1] == titanic_sub_new.shape[1]), "Not Equal"
#
#
# # In[33]:
#
#
# # Generate the predictions
# submission = tpot.predict(titanic_sub_new)
#
#
# # In[34]:
#
#
# # Create the submission file
# final = pd.DataFrame({'PassengerId': titanic_sub['PassengerId'], 'Survived': submission})
# final.to_csv('data/submission.csv', index = False)


# In[35]:


# final.shape


# There we go! We have successfully generated the predictions for the 418 data points in the submission dataset, and we're good to go ahead to submit these predictions on Kaggle.

# # XGBOOST Experiment

# In[23]:


import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from src.hypersearch import xgbc_random
from config.xgboost import config_dict


# In[24]:


print(xgbc_random(
    config_dict,
    3600,
    titanic_new[training_indices],
    titanic_class[training_indices],
    titanic_new[validation_indices],
    titanic.loc[validation_indices, 'class'].values))


# In[18]:


# import imp
# imp.reload(h2o)


# In[1]:


# import h2o
# from h2o.automl import H2OAutoML
#
#
# # In[2]:
#
#
# h2o.init(ip='192.168.1.191')
#
#
# # In[34]:
#
#
# train = h2o.import_file("data/titanic_train.csv")
# test = h2o.import_file("data/titanic_test.csv")
#
#
# # In[35]:
#
#
# train = train[:, ["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]]
#
#
# # In[36]:
#
#
# train['Sex'] = (train['Sex'] == 'male').ifelse(0, train["Sex"])
# train["Sex"] = (train['Sex'] == 'female').ifelse(1, train["Sex"])
# train["Embarked"] = (train['Embarked'] == 'S').ifelse(0, train["Embarked"])
# train["Embarked"] = (train['Embarked'] == 'C').ifelse(1, train["Embarked"])
# train["Embarked"] = (train['Embarked'] == 'Q').ifelse(2, train["Embarked"])
#
#
# # In[37]:
#
#
# x = train.columns
# y = "Survived"
# x.remove(y)
#
#
# # In[38]:
#
#
# train[y] = train[y].asfactor()
#
#
# # In[39]:
#
#
# train["Sex"] = train["Sex"].asfactor()
# train["Embarked"] = train["Embarked"].asfactor()
# train["Pclass"] = train["Pclass"].asfactor()
#
#
# # In[40]:
#
#
# aml = H2OAutoML(max_runtime_secs=3600)
# aml.train(x=x, y=y, training_frame=train)
#
