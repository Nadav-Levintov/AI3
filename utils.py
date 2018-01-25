import math
from sklearn.metrics import accuracy_score
import numpy as np

from sklearn.model_selection import StratifiedKFold



def split2LearningAndValidation(samples, y,learning_ratio):
    num_samples = len(samples)
    num_learning = math.floor(learning_ratio * num_samples)
    num_validation = math.ceil((1-learning_ratio) * num_samples)
    return samples[0:num_learning ], y[0:num_learning], samples[num_learning: len(samples)], y[num_learning : len(samples)]


def score(clf, X, y):
    X_arr = np.array(X)
    y_arr = np.array(y)

    skf = StratifiedKFold(4)
    avg = 0
    for train, test in skf.split(X_arr, y_arr):
        clf = clf.fit(X_arr[train],y_arr[train])
        y_pred = clf.predict(X_arr[test])
        avg += accuracy_score(y_arr[test],y_pred)

    return avg/4




# def score(clf,X,y):
#     k = 4
#     k_grps = kFoldCrossVlidationSplit(k,X,y)
#     cv_grps = train_test_split(X)
#     cross_validate()
#
#
#     avg_score = 0.0
#     for grp in k_grps:
#         curr_score = 0.0
#         validation_grp = grp[0]
#         validation_labels = grp[1]
#
#         learning_grp = []
#         learning_labels = []
#         for grp_2 in k_grps:
#             if grp_2 != grp:
#                 learning_grp += grp_2[0]
#                 learning_labels += grp_2[1]
#         curr_clf = copy.copy(clf)
#         curr_clf.fit(learning_grp,learning_labels)
#         predict_vec = curr_clf.predict(validation_grp)
#         hit_count = 0
#         for i in range(len(validation_labels)):
#             if predict_vec[i] == validation_labels[i]:
#                 hit_count+=1
#         curr_score =  hit_count/len(validation_labels)
#         #print("curr score " + str(curr_score))
#         avg_score+=curr_score*(1/k)
#     #print("avg: " + str(avg_score))
#     return avg_score

# a = kFoldCrossVlidationSplit(4, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
# for b in a:
#     print (b)