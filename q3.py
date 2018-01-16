import copy
import csv
import math

import numpy as np
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
sorted_flare_list = copy.deepcopy(flare_list)

for i in range(len(flare_list[0]) - 2, -1 , -1):
    sorted_flare_list = sorted (sorted_flare_list, key=lambda x:x[i])

X = sorted_flare_list
y= []
for line in X:
    y.append(line.pop())

X_arr = np.array(X)

y_arr = np.array(y)


#id3
clf = DecisionTreeClassifier(criterion="entropy")

range_three_quarters_of_arr = range(math.ceil(3 * len(flare_list) / 4))
range_last_quarter = range(math.ceil(3 * len(flare_list) / 4), len(flare_list))
clf = clf.fit(X_arr[range_three_quarters_of_arr], y_arr[range_three_quarters_of_arr])
#print("test")
#y_pred = clf.predict(X_arr[range_last_quarter])
#print(accuracy_score(y_arr[range_last_quarter], y_pred))
#print("train")
y_pred = clf.predict(X_arr[range_three_quarters_of_arr])
print(accuracy_score(y_arr[range_three_quarters_of_arr], y_pred))
##########################################################
#Now we will do under fit by only learning on 1 feature
weak_feature_list = []
for f in flare_list:
    weak_feature_list.append([f[0],f[len(f)-1]])

X = weak_feature_list
y= []
for line in X:
    y.append(line.pop())

X_arr = np.array(X)
y_arr = np.array(y)

#id3
clf = DecisionTreeClassifier(criterion="entropy")

clf = clf.fit(X_arr[range_three_quarters_of_arr], y_arr[range_three_quarters_of_arr])
y_pred = clf.predict(X_arr[range_last_quarter])

#print("test")
#y_pred = clf.predict(X_arr[range_last_quarter])
#print(accuracy_score(y_arr[range_last_quarter], y_pred))
#print("train")
y_pred = clf.predict(X_arr[range_three_quarters_of_arr])
print(accuracy_score(y_arr[range_three_quarters_of_arr], y_pred))

