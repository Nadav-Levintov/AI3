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

count = 0
for i in range(len(data_wo_header)):
    if learning_full[i] != data_wo_header[i]:
        print("missmatch")
        count+=1
if(count>0):
    print("mismatches " , count , " out of " , len(data_wo_header))
    exit(0)

knn_classifier = KNeighborsClassifier(n_neighbors = 5)
knn_classifier.fit(learning_grp,learning_labels)
classify_vec = knn_classifier.predict(validation_grp)
if len(classify_vec) != len(validation_labels):
    print("length mismatch")
    exit(0)
hit_count = 0;
for i in range(len(classify_vec)):
    if classify_vec[i] == validation_labels[i]:
        hit_count+=1
print("precision of KNN without chosing features: " + str(hit_count/len(validation_labels)))

knn_classifier_for_sfs = KNeighborsClassifier(n_neighbors = 5)
b = sfs(learning_grp,learning_labels,8,knn_classifier_for_sfs,score)
print(b)

knn_classifier_for_sfs_p = copy.copy(knn_classifier_for_sfs)
knn_classifier_for_sfs_p.fit([[obj[feature] for feature in b] for obj in learning_grp ],learning_labels)
validation_vec = knn_classifier_for_sfs_p.predict([[obj[feature] for feature in b] for obj in validation_grp ])
hit_count = 0
for i in range(len(validation_labels)):
    if validation_labels[i] == validation_vec[i]:
        hit_count+=1
print("knn = 5 , b = 8 precision is " + str(hit_count/len(validation_grp)))

# for i in range(len(data)):
#     print (data[i][0:3] )


