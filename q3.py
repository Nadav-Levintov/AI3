import copy
import csv
import math

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

flare_list = []
with open('flare.csv', 'r') as flare_file:
    flare_reader = csv.reader(flare_file, delimiter=',', quotechar='|')
    for row in flare_reader:
        flare_list.append(row)

features = flare_list[0]
features.pop()

#over fitting - lets sort the input


flare_list.remove(flare_list[0])

X = copy.deepcopy(flare_list)
y= []
for line in X:
    y.append(line.pop())

X_arr = np.array(X)
y_arr = np.array(y)



clf = DecisionTreeClassifier(criterion="entropy")
skf = StratifiedKFold(4)
train_accuracy =0
test_accuracy =0
for train, test in skf.split(X_arr, y_arr):
    clf = clf.fit(X_arr[train],y_arr[train])
    y_pred_train = clf.predict(X_arr[train])
    y_pred_test = clf.predict(X_arr[test])
    train_accuracy += accuracy_score(y_arr[train], y_pred_train)
    test_accuracy += accuracy_score(y_arr[test], y_pred_test)

train_accuracy /= 4
test_accuracy /= 4



#train
print(train_accuracy)
#test
#print(test_accuracy)
##########################################################
#Now we will do under fit by only learning on 1 feature
X = copy.deepcopy(flare_list)
y= []
for line in X:
    y.append(line.pop())

X_arr = np.array(X)
y_arr = np.array(y)

clf = DecisionTreeClassifier(criterion="entropy", max_features=1, max_depth=1)

skf = StratifiedKFold(4)
train_accuracy =0
test_accuracy =0
for train, test in skf.split(X_arr, y_arr):
    clf = clf.fit(X_arr[train],y_arr[train])
    y_pred_train = clf.predict(X_arr[train])
    y_pred_test = clf.predict(X_arr[test])
    train_accuracy += accuracy_score(y_arr[train], y_pred_train)
    test_accuracy += accuracy_score(y_arr[test], y_pred_test)

train_accuracy /= 4
test_accuracy /= 4

#train
print(train_accuracy)
#test
#print(test_accuracy)

