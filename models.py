import numpy as np
import pandas as pd

data = pd.read_csv('instagram/data.csv')
# print(data.head())
#Creating relevant features list
imp_list=['nums/length username','private','#posts','#followers','#follows','fake']
#Creating Genuine users dataset
data.drop(data.columns.difference(imp_list), axis = 1, inplace = True)
colremap = {
    'nums/length username': 'n_by_len_uname',
    'private': 'private',
    '#posts': 'posts',
    '#followers': 'followers',
    '#follows': 'follows',
    'fake': 'fake'
}
data.rename(columns=colremap, inplace=True)
# print(data.head())
data.query('fake == 1').to_csv('instagram/fake.csv',index=False)
data.query('fake == 0').to_csv('instagram/real.csv',index=False)

data = pd.read_csv('facebook/data.csv')
# print(data.head())
#Creating relevant features list
imp_list=['#friends','#following','#community','#postshared','avgcomment/post','likes/post','#tags/post','Label']
#Creating Genuine users dataset
data.drop(data.columns.difference(imp_list), axis = 1, inplace = True)
# print(data.head())
data.query('Label == 1').to_csv('facebook/fake.csv',index=False)
data.query('Label == 0').to_csv('facebook/real.csv',index=False)

from sklearn.tree import DecisionTreeClassifier           # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split      # FOR train_test_split function
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('instagram/data.csv')
features=['nums/length username','private','#posts','#followers','#follows']

#split dataset in features and target variable
X = dataset[features] # Features
y = dataset['fake'] # Target variable

#     We have X_train, y_train, X_test, y_test.
#     Using these lists and dataframes we will randomly create two non-overlapping datasets 
#         1. training set
#         2. testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=104,test_size=0.25, shuffle=True)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(min_impurity_decrease=0.001)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_predict = clf.predict(X_test)

# print(y_predict)

conf_matrix = confusion_matrix(y_test, y_predict)

#true_negative
TN = conf_matrix[0][0]
#false_negative
FN = conf_matrix[1][0]
#false_positive
FP = conf_matrix[0][1]
#true_positive
TP = conf_matrix[1][1]

# Recall is the ratio of the total number of correctly classified positive examples divided by the total number of positive examples. 
# High Recall indicates the class is correctly recognized (small number of FN)

recall = (TP)/(TP + FN)

# Precision is the the total number of correctly classified positive examples divided by the total number of predicted positive examples. 
# High Precision indicates an example labeled as positive is indeed positive (small number of FP)

precision = (TP)/(TP + FP)

fmeasure = (2*recall*precision)/(recall+precision)
accuracy = (TP + TN)/(TN + FN + FP + TP)

accuracy_score(y_test, y_predict)

print("------ CLASSIFICATION PERFORMANCE OF DECISION TREE MODEL (Instagram) ------ \n"\
      "\n Recall : ", (recall*100) ,"%" \
      "\n Precision : ", (precision*100) ,"%" \
      "\n Accuracy : ", (accuracy*100) ,"%" \
      "\n F-measure : ", (fmeasure*100) ,"%" )

import pickle
with open('instagram/pickleOutput', 'wb') as f:
    pickle.dump(clf, f)

with open('instagram/pickleOutput', 'rb') as f:
    mp = pickle.load(f)
    
pickleTest = mp.predict(X_test)
# print(pickleTest == y_predict)

dataset = pd.read_csv('facebook/data.csv')
features=['#friends','#following','#community','#postshared','avgcomment/post','likes/post','#tags/post']

#split dataset in features and target variable
X = dataset[features] # Features
y = dataset['Label'] # Target variable

#     We have X_train, y_train, X_test, y_test.
#     Using these lists and dataframes we will randomly create two non-overlapping datasets 
#         1. training set
#         2. testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=104,test_size=0.25, shuffle=True)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(min_impurity_decrease=0.001)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_predict = clf.predict(X_test)

# print(y_predict)

conf_matrix = confusion_matrix(y_test, y_predict)

#true_negative
TN = conf_matrix[0][0]
#false_negative
FN = conf_matrix[1][0]
#false_positive
FP = conf_matrix[0][1]
#true_positive
TP = conf_matrix[1][1]

# Recall is the ratio of the total number of correctly classified positive examples divided by the total number of positive examples. 
# High Recall indicates the class is correctly recognized (small number of FN)

recall = (TP)/(TP + FN)

# Precision is the the total number of correctly classified positive examples divided by the total number of predicted positive examples. 
# High Precision indicates an example labeled as positive is indeed positive (small number of FP)

precision = (TP)/(TP + FP)

fmeasure = (2*recall*precision)/(recall+precision)
accuracy = (TP + TN)/(TN + FN + FP + TP)

accuracy_score(y_test, y_predict)

print("------ CLASSIFICATION PERFORMANCE OF DECISION TREE MODEL (Facebook) ------ \n"\
      "\n Recall : ", (recall*100) ,"%" \
      "\n Precision : ", (precision*100) ,"%" \
      "\n Accuracy : ", (accuracy*100) ,"%" \
      "\n F-measure : ", (fmeasure*100) ,"%" )

import pickle
with open('facebook/pickleOutput', 'wb') as f:
    pickle.dump(clf, f)

with open('facebook/pickleOutput', 'rb') as f:
    mp = pickle.load(f)
    
pickleTest = mp.predict(X_test)
# print(pickleTest == y_predict)

dataset = pd.read_csv('twitter/data.csv')

#split dataset in features and target variable
X = dataset.drop(columns = ['label']) # Features
y = dataset['label'] # Target variable

#     We have X_train, y_train, X_test, y_test.
#     Using these lists and dataframes we will randomly create two non-overlapping datasets 
#         1. training set
#         2. testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=104,test_size=0.25, shuffle=True)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(min_impurity_decrease=0.001)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_predict = clf.predict(X_test)

# print(y_predict)

conf_matrix = confusion_matrix(y_test, y_predict)

#true_negative
TN = conf_matrix[0][0]
#false_negative
FN = conf_matrix[1][0]
#false_positive
FP = conf_matrix[0][1]
#true_positive
TP = conf_matrix[1][1]

# Recall is the ratio of the total number of correctly classified positive examples divided by the total number of positive examples. 
# High Recall indicates the class is correctly recognized (small number of FN)

recall = (TP)/(TP + FN)

# Precision is the the total number of correctly classified positive examples divided by the total number of predicted positive examples. 
# High Precision indicates an example labeled as positive is indeed positive (small number of FP)

precision = (TP)/(TP + FP)

fmeasure = (2*recall*precision)/(recall+precision)
accuracy = (TP + TN)/(TN + FN + FP + TP)

accuracy_score(y_test, y_predict)

print("------ CLASSIFICATION PERFORMANCE OF DECISION TREE MODEL (Twitter) ------ \n"\
      "\n Recall : ", (recall*100) ,"%" \
      "\n Precision : ", (precision*100) ,"%" \
      "\n Accuracy : ", (accuracy*100) ,"%" \
      "\n F-measure : ", (fmeasure*100) ,"%" )

import pickle
with open('twitter/pickleOutputNew', 'wb') as f:
    pickle.dump(clf, f)

with open('twitter/pickleOutputNew', 'rb') as f:
    mp = pickle.load(f)
    
pickleTest = mp.predict(X_test)
# print(pickleTest == y_predict)