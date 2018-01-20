import copy
import csv

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sfs import split2LearningAndValidation

flare_list = []
with open('flare.csv', 'r') as flare_file:
    flare_reader = csv.reader(flare_file, delimiter=',', quotechar='|')
    for row in flare_reader:
        flare_list.append(row)

features = flare_list[0]
features.pop()

flare_list.remove(flare_list[0])

labels= []
for line in flare_list:
    labels.append(line.pop())

X , y , X_test , y_test = split2LearningAndValidation(copy.deepcopy(flare_list),copy.deepcopy(labels),0.75)

#No pruning
clf = DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X,y)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


print(accuracy)

#pre pruning

X , y , X_test , y_test = split2LearningAndValidation(copy.deepcopy(flare_list),copy.deepcopy(labels),0.75)

clf = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=20)
clf = clf.fit(X,y)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)