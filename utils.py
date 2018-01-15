import math

def split2LearningAndValidation(samples, y,learning_ratio):
    num_samples = len(samples)
    num_learning = math.floor(learning_ratio * num_samples)
    num_validation = math.ceil((1-learning_ratio) * num_samples)
    return samples[0:num_learning ], y[0:num_learning], samples[num_learning: len(samples)], y[num_learning : len(samples)]

def score(clf,X,y):
    learning_grp, learning_labels, validation_grp, validation_labels = split2LearningAndValidation(X, y, 0.75)
    clf.fit(learning_grp,learning_labels)
    predict_vec = clf.predict(validation_grp)
    hit_count = 0
    for i in range(len(validation_labels)):
        if predict_vec[i] == validation_labels[i]:
            hit_count+=1
    return hit_count/len(validation_labels)