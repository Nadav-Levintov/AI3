import csv

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

flare_list = []
with open('flare.csv', 'r') as flare_file:
    flare_reader = csv.reader(flare_file, delimiter=',', quotechar='|')
    for row in flare_reader:
        flare_list.append(row)

features = flare_list[0]
features.pop()

flare_list.remove(flare_list[0])

X = flare_list
y= []
for line in X:
    y.append(line.pop())

X_arr = np.array(X)
y_arr = np.array(y)

skf = StratifiedKFold(4)
#No pruning
clf = DecisionTreeClassifier(criterion="entropy")
avg = 0

for train, test in skf.split(X_arr, y_arr):
    clf = clf.fit(X_arr[train],y_arr[train])
    y_pred = clf.predict(X_arr[test])
    avg += accuracy_score(y_arr[test],y_pred)


print((avg/4))

#pre pruning
clf = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=20)
avg = 0

for train, test in skf.split(X_arr, y_arr):
    clf = clf.fit(X_arr[train],y_arr[train])
    y_pred = clf.predict(X_arr[test])
    avg += accuracy_score(y_arr[test],y_pred)


print((avg/4))