# Avoiding warning
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn
# _______________________________

# Essential Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# _____________________________

# np.random.seed(seed=111)

# scikit-learn :
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, \
    RandomForestClassifier, \
    AdaBoostClassifier, \
    GradientBoostingClassifier, \
    ExtraTreesClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

Names = ['LR', 'KNN', 'DT', 'NB', 'Bagging', 'RF', 'AB', 'GB', 'SVM', 'LDA', 'ET']

Classifiers = [
    LogisticRegression(),           #1
    KNeighborsClassifier(),         #2
    DecisionTreeClassifier(),       #3
    GaussianNB(),                   #4
    BaggingClassifier(),            #5
    RandomForestClassifier(),       #6
    AdaBoostClassifier(),           #7
    GradientBoostingClassifier(),   #8
    SVC(probability=True),          #9
    LinearDiscriminantAnalysis(),   #10
    # ExtraTreesClassifier(),         #11
]

from sklearn.metrics import accuracy_score, \
        log_loss, \
        classification_report, \
        confusion_matrix, \
        roc_auc_score,\
        average_precision_score,\
        auc,\
        roc_curve, f1_score, recall_score, matthews_corrcoef, auc


def testModel(X_test, y_test):

    ### Retrieve Model with joblib ###
    from sklearn.externals import joblib
    with open('dumpModel.pkl', 'rb') as File:
        model = joblib.load(File)

    # yHat = model.predict(X_test)

    y_proba = model.predict_proba(X_test)[:, 1]
    ##########################################
    # print(FPR)
    # print(TPR)
    ##########################################
    y_artificial = model.predict(X_test)

    print('Accuracy: {0:.2f}%'.format(accuracy_score(y_pred=y_artificial, y_true=y_test)*100.0))
    print('auROC: {0:.4f}'.format(roc_auc_score(y_true=y_test, y_score=y_proba)))
    print('auPR: {0:.4f}'.format(average_precision_score(y_true=y_test, y_score=y_proba)))  # auPR
    print('F1 Score: {0:.4f}'.format(f1_score(y_true=y_test, y_pred=y_artificial)))
    print('MCC: {0:.2f}'.format(matthews_corrcoef(y_true=y_test, y_pred=y_artificial)))

    CM = confusion_matrix(y_pred=y_artificial, y_true=y_test)

    TN, FP, FN, TP = CM.ravel()
    print('Sensitivity: {0:.2f}%'.format(((TP) / (TP + FN)) * 100.0))
    print('Specificity: {0:.2f}%'.format(((TN) / (TN + FP)) * 100.0))
    print('Confusion Matrix:')
    print(CM)



