from sfs import sfs
from sklearn.neighbors import KNeighborsClassifier
import csv
from utils import split2LearningAndValidation, score
import copy
import sys




file_name = 'flare.csv'
with open(file_name, 'rU') as f:
    reader = csv.reader(f)
    data = list(list(rec) for rec in csv.reader(f, delimiter=','))
    f.close() #close the csv

header = data[0]
# print(len(header))
# print(header)
data_wo_header = data[1:]
data_wo_header_class = [x[0:32] for x in data_wo_header]
labels = [x[32] for x in data_wo_header]
# print(labels)
# for row in data_wo_header_class:
#     print(row)
learning_grp , learning_labels , validation_grp , validation_labels = split2LearningAndValidation(data_wo_header_class,labels,0.75)

learning_full = []
for i in range(len(learning_grp)):
    learning_full.append(learning_grp[i] + [learning_labels[i]])
for i in range(len(validation_grp)):
    learning_full.append(validation_grp[i] +[validation_labels[i]])



knn_classifier = KNeighborsClassifier(n_neighbors = 5)
knn_classifier.fit(learning_grp,learning_labels)
classify_vec = knn_classifier.predict(validation_grp)

hit_count = 0;
for i in range(len(classify_vec)):
    if classify_vec[i] == validation_labels[i]:
        hit_count+=1
#"precision of KNN without choosing features: " +
print(str(hit_count/len(validation_labels)))

knn_classifier_for_sfs = KNeighborsClassifier(n_neighbors = 5)
b = sfs(learning_grp,learning_labels,8,knn_classifier_for_sfs,score)
#print(b)

knn_classifier_for_sfs_p = copy.copy(knn_classifier_for_sfs)
knn_classifier_for_sfs_p.fit([[obj[feature] for feature in b] for obj in learning_grp ],learning_labels)
validation_vec = knn_classifier_for_sfs_p.predict([[obj[feature] for feature in b] for obj in validation_grp ])
hit_count = 0
for i in range(len(validation_labels)):
    if validation_labels[i] == validation_vec[i]:
        hit_count+=1
#"knn = 5 , b = 8 precision is " +
print( str(hit_count/len(validation_grp)))

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
