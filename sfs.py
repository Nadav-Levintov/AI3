import math
import copy



def split2LearningAndValidation(samples, y,learning_ratio):
    num_samples = len(samples)
    num_learning = math.floor(learning_ratio * num_samples)
    num_validation = math.ceil((1-learning_ratio) * num_samples)
    return samples[0:num_learning ], y[0:num_learning], samples[num_learning: len(samples)], y[num_learning : len(samples)]


def sfs(x, y, k, clf, score):
    """
    :param x: feature set to be trained using clf. list of lists.
    :param y: labels corresponding to x. list.
    :param k: number of features to select. int
    :param clf: classifier to be trained on the feature subset.
    :param score: utility function for the algorithm, that receives clf, feature subset and labeles, returns a score. 
    :return: list of chosen feature indexes
    """
    num_features = len(x[0])
    featureSubSet = []
    init_score = 0

    while(len(featureSubSet) < k):
        best_score = init_score
        for f in range(num_features):

            learning_grp , learning_labels , validation_grp , validation_labels = split2LearningAndValidation(x,y,1)
            temp_featureSubSet = list(featureSubSet) #copy
            if(f not in featureSubSet):
                temp_featureSubSet.append(f)
            else:
                continue
            #with the new feature : learn and get score
            temp_clf = copy.copy(clf)
            curr_score =  score(temp_clf, [[obj[feature] for feature in temp_featureSubSet] for obj in learning_grp ] ,learning_labels)
            if(curr_score > best_score):
                #TODO : what happens where each additional feature to the subset is making the score worst before reacking K ??
                #TODO : probably dont need to split x into learning and validation. score function is. for now ratio = 1.
                best_score = curr_score
                featureSubSet = list(temp_featureSubSet) #copy
    return featureSubSet



    # print(learning_grp)
    # print(validation_grp)


#sfs([1,2,3,4,5,6,7,8,9],1,2,3,4)