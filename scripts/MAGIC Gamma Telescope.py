
# coding: utf-8

# # MAGIC Gamma Telescope - TPOT Classification Study

# The below gives information about the data set:

# The data are MC generated (see below) to simulate registration of high energy gamma particles in a ground-based atmospheric Cherenkov gamma telescope using the imaging technique. Cherenkov gamma telescope observes high energy gamma rays, taking advantage of the radiation emitted by charged particles produced inside the electromagnetic showers initiated by the gammas, and developing in the atmosphere. This Cherenkov radiation (of visible to UV wavelengths) leaks through the atmosphere and gets recorded in the detector, allowing reconstruction of the shower parameters. The available information consists of pulses left by the incoming Cherenkov photons on the photomultiplier tubes, arranged in a plane, the camera. Depending on the energy of the primary gamma, a total of few hundreds to some 10000 Cherenkov photons get collected, in patterns (called the shower image), allowing to discriminate statistically those caused by primary gammas (signal) from the images of hadronic showers initiated by cosmic rays in the upper atmosphere (background).
#
# Typically, the image of a shower after some pre-processing is an elongated cluster. Its long axis is oriented towards the camera center if the shower axis is parallel to the telescope's optical axis, i.e. if the telescope axis is directed towards a point source. A principal component analysis is performed in the camera plane, which results in a correlation axis and defines an ellipse. If the depositions were distributed as a bivariate Gaussian, this would be an equidensity ellipse. The characteristic parameters of this ellipse (often called Hillas parameters) are among the image parameters that can be used for discrimination. The energy depositions are typically asymmetric along the major axis, and this asymmetry can also be used in discrimination. There are, in addition, further discriminating characteristics, like the extent of the cluster in the image plane, or the total sum of depositions.
#
# https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope
#

# Attribute Information:
#
# 1. fLength: continuous # major axis of ellipse [mm]
# 2. fWidth: continuous # minor axis of ellipse [mm]
# 3. fSize: continuous # 10-log of sum of content of all pixels [in #phot]
# 4. fConc: continuous # ratio of sum of two highest pixels over fSize [ratio]
# 5. fConc1: continuous # ratio of highest pixel over fSize [ratio]
# 6. fAsym: continuous # distance from highest pixel to center, projected onto major axis [mm]
# 7. fM3Long: continuous # 3rd root of third moment along major axis [mm]
# 8. fM3Trans: continuous # 3rd root of third moment along minor axis [mm]
# 9. fAlpha: continuous # angle of major axis with vector to origin [deg]
# 10. fDist: continuous # distance from origin to center of ellipse [mm]
# 11. class: g,h # gamma (signal), hadron (background)
#
# g = gamma (signal): 12332
# h = hadron (background): 6688
#
# For technical reasons, the number of h events is underestimated. In the real data, the h class represents the majority of the events.
#
# The simple classification accuracy is not meaningful for this data, since classifying a background event as signal is worse than classifying a signal event as background. For comparison of different classifiers an ROC curve has to be used. The relevant points on this curve are those, where the probability of accepting a background event as signal is below one of the following thresholds: 0.01, 0.02, 0.05, 0.1, 0.2 depending on the required quality of the sample of the accepted events for different experiments.

# In[1]:


# Import required libraries
from tpot import TPOTClassifier
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np


# In[2]:


#Load the data
telescope=pd.read_csv('data/MAGIC Gamma Telescope Data.csv')
# telescope.head(5)


# As can be seen in the above the class object is organized here, and hence for better results, I start with randomly shuffling the data.

# In[3]:


telescope_shuffle=telescope.iloc[np.random.permutation(len(telescope))]
# telescope_shuffle.head()


# Above also rearranges the index number and below I basically reset the Index Numbers.

# In[4]:


tele=telescope_shuffle.reset_index(drop=True)
# tele.head()


# # Pre-Processing Data

# In[5]:


# Check the Data Type
# tele.dtypes


# Class is a Categorical Variable as can be seen from above. The levels in it are:

# In[6]:


# for cat in ['Class']:
#     print("Levels for catgeory '{0}': {1}".format(cat, tele[cat].unique()))


# Next, the categorical variable is numerically encoded since TPOT for now cannot handle categorical variables. 'g' is coded as 0 and 'h' as 1.

# In[7]:


tele['Class']=tele['Class'].map({'g':0,'h':1})


# A check is performed to see if there are any missing values and code then accordingly.

# In[8]:


tele = tele.fillna(-999)
# pd.isnull(tele).any()


# In[9]:


# tele.shape


# Finally we store the class labels, which we need to predict, in a separate variable.

# In[10]:


tele_class = tele['Class'].values


# # Data Analysis using TPOT

# To begin our analysis, we need to divide our training data into training and validation sets. The validation set is just to give us an idea of the test set error.

# In[11]:


training_indices, validation_indices = training_indices, testing_indices = train_test_split(tele.index, stratify = tele_class, train_size=0.75, test_size=0.25)
# training_indices.size, validation_indices.size


# After that, we proceed to calling the `fit()`, `score()` and `export()` functions on our training dataset.
# An important TPOT parameter to set is the number of generations (via the `generations` kwarg). Since our aim is to just illustrate the use of TPOT, we assume the default setting of 100 generations, whilst bounding the total running time via the `max_time_mins` kwarg (which may, essentially, override the former setting). Further, we enable control for the maximum amount of time allowed for optimization of a single pipeline, via `max_eval_time_mins`.
#
# On a standard laptop with 4GB RAM, each generation takes approximately 5 minutes to run. Thus, for the default value of 100, without the explicit duration bound, the total run time could be roughly around 8 hours.

# In[12]:


from config.classifier_models_only import classifier_config_dict
time_allocated = 60


# In[13]:


tpot = TPOTClassifier(
    verbosity=3,
    max_time_mins=time_allocated,
    config_dict=classifier_config_dict,
    scoring="neg_log_loss",
    n_jobs=8)
tpot.fit(tele.drop('Class',axis=1).loc[training_indices].values, tele.loc[training_indices,'Class'].values)


# In the above, 7 generations were computed, each giving the training efficiency of fitting model on the training set. As evident, the best pipeline is the one that has the CV score of 85.335%. The model that produces this result is pipeline, consisting of a logistic regression that adds synthetic features to the input data, which then get utilized by a decision tree classifier to form the final predictions.
#
# Next, the test error is computed for validation purposes.

# In[13]:


tpot.score(tele.drop('Class',axis=1).loc[validation_indices].values, tele.loc[validation_indices, 'Class'].values)


# As can be seen, the test accuracy is 85.573%.

# In[14]:


# tpot.export('tpot_MAGIC_Gamma_Telescope_pipeline.py')


# In[ ]:


# %load tpot_MAGIC_Gamma_Telescope_pipeline.py
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline, make_union
# from sklearn.tree import DecisionTreeClassifier
# from tpot.builtins import StackingEstimator
#
# # NOTE: Make sure that the class is labeled 'target' in the data file
# tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
# features = tpot_data.drop('target', axis=1).values
# training_features, testing_features, training_target, testing_target =             train_test_split(features, tpot_data['target'].values, random_state=42)
#
# # Score on the training set was:0.853347788745
# exported_pipeline = make_pipeline(
#     StackingEstimator(estimator=LogisticRegression(C=10.0, dual=False, penalty="l2")),
#     DecisionTreeClassifier(criterion="gini", max_depth=7, min_samples_leaf=5, min_samples_split=7)
# )
#
# exported_pipeline.fit(training_features, training_target)
# results = exported_pipeline.predict(testing_features)


# In[14]:


import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from src.hypersearch import xgbc_random
from config.xgboost import config_dict


# In[16]:


print(xgbc_random(
    config_dict,
    3600,
    tele.drop('Class',axis=1).loc[training_indices].values,
    tele.loc[training_indices,'Class'].values,
    tele.drop('Class',axis=1).loc[validation_indices].values,
    tele.loc[validation_indices, 'Class'].values))
