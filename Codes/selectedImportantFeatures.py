import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import save

def selectKImportance(model, X):

    importantFeatures = model.feature_importances_
    Values = np.sort(importantFeatures)[::-1] #SORTED

    K = importantFeatures.argsort()[::-1][:len(Values[Values>0.00])]
    save.saveBestK(K)

    # print(' --- begin --- ')
    #
    # for i in K:
    #     print(i, end=', ')
    # print()
    # print(' --- end dumping webserver (425) --- ')
    #
    # C=1
    # for value, eachk in zip(Values, K):
    #     print('rank:{}, value:{}, index:({})'.format(C, value, eachk))
    #
    #      C += 1
    # print('--- end ---')

    ##############################
    # print(Values)
    # print()
    # print(Values[Values>0.00])
    ##############################

    return X[:, K]


def importantFeatures(X, Y):

    model = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=500, learning_rate=1.0)
    model.fit(X, Y)

    return selectKImportance(model, X)



