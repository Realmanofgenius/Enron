
# coding: utf-8

# In[35]:


# %load poi_id.py
#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import numpy as np
import pandas as pd

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 
                 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 
                 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi',
                 'poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    enron_dataset = pickle.load(data_file)
    
df = pd.DataFrame.from_dict(enron_dataset, orient='index')
df = df.replace('NaN', np.nan)

### Task 2: Remove outliers
enron_dataset.pop('THE TRAVEL AGENCY IN THE PARK')
enron_dataset.pop('TOTAL')
enron_dataset.pop('LOCKHART EUGENE E')

### Task 3: Create new feature(s)
### **NEW FEATURE WAS NOT USED IN DATASET**COMMENTED OUT**
### count = 0
###for i in enron_dataset:
   ### enron_dataset[i]['fraction_poi_emails'] = fraction_poi_emails[count]
    ### count += 1
    
### Store to my_dataset for easy export below.
my_dataset = enron_dataset


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.4, random_state = 42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

###StratifiedShuffleSplit for Validation
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.1, random_state=42)

sss = StratifiedShuffleSplit(labels, 200, test_size=0.1, random_state=42)

###DecisionTree

#param_grid = {
 #        'min_samples_split': [2],
  #       'max_depth': [1],
   #      'max_features': [3, 10]
    #      }
#clf = GridSearchCV(DecisionTreeClassifier(), param_grid)
#clf = clf.fit(features_train, labels_train)

###Setting up parameters based on GridSearchCV results from Enron_Final1.ipynb
pipe = Pipeline(steps=[('min_max_scaler', MinMaxScaler()),
                       #('logistic_regression', LogisticRegression()),
                       ('tree', DecisionTreeClassifier()),
                      ])

parameters = dict(tree__max_features = [3],
                  tree__min_samples_split = [2],
                  tree__criterion = ['gini']
                 )

grid_search = GridSearchCV(pipe, parameters, n_jobs = 1, cv = sss, scoring='f1', verbose = 2)

grid_search.fit(features, labels)

print ''
print grid_search.best_estimator_

###LogisticRegression
#param_grid = {'C': [0.001] }
#clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
#clf = clf.fit(features_train, labels_train)

###Printing best parameters
print ''
print 'Best score: %0.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
       print '\t%s: %r' % (param_name, best_parameters[param_name])


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

