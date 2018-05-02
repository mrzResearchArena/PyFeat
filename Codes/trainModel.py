### Avoid warning ###
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn


### Essential ###
import pandas as pd
import numpy as np


### Fixed State ###
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



def train(args):

    ### Load Dataset ###
    D = pd.read_csv(args.dataset, header=None)

    ### Splitting dataset into X, Y ###
    X_train = D.iloc[:, :-1].values
    Y_train = D.iloc[:, -1].values


    ### Random Shuffle ###
    from sklearn.utils import shuffle
    X_train, Y_train = shuffle(X_train, Y_train)  # Avoiding bias


    ## Scaling using sci-kit learn ###
    from sklearn.preprocessing import StandardScaler
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)


    ### Run Machine Learning Model ###
    print('Model training using ', end='')

    #1. using LR:
    if args.model == 'LR':
        model = LogisticRegression()
        print('LR ', end='')

    #2. using KNN:
    if args.model == 'KNN':
        model = KNeighborsClassifier(n_neighbors=args.K)
        print('KNN ', end='')

    #3. using DT:
    if args.model == 'DT':
        model = DecisionTreeClassifier()
        print('DT ', end='')

    #4. using SVM:
    if args.model == 'SVM':
        model = SVC(probability=True)
        print('SVM ', end='')


    #5. using NB:
    if args.model == 'NB':
        model = GaussianNB()
        print('NB ',end='')

    #6. using Bagging:
    if args.model == 'Bagging':
        model = BaggingClassifier()
        print('Bagging ', end='')

    #7. using RF:
    if args.model == 'RF':
        model = RandomForestClassifier()
        print('RF ', end='')

    #8. using AB:
    if args.model == 'AB':
        model = AdaBoostClassifier()
        print('AB' , end='')

    #9. using GB:
    if args.model == 'GB':
        model = GradientBoostingClassifier()
        print('GB ', end='')

    #10. using LDA:
    if args.model == 'LDA':
        model = LinearDiscriminantAnalysis()
        print('LDA ', end='')

    '''
    You can use any model instead of logistics regression,
    and an even different parameter.
    
    Suggested: GridSearchCV
    (http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
    '''

    ### Fitting Model ###
    model.fit(X_train, Y_train)
    ### --- ###


    ### Dump Model with joblib ###
    from sklearn.externals import joblib
    with open('dumpModel.pkl', 'wb') as File:
        joblib.dump(model, File)


    ### Dump Model with pickle ###
    # import pickle
    # with open('modelDump.pkl', 'wb') as pickleFile:
    #     pickle.dump(model, pickleFile)
    ##############################

    print('classifier.')


import argparse

p = argparse.ArgumentParser(description='Training the model with LR.')
p.add_argument('-data', '--dataset', type=str, help='~/dataset.csv', default='optimumDataset.csv')

# p.add_argument('-lr', '--LR', type=int, help='using Logistics Regression', default=0, choices=[0, 1])
# p.add_argument('-svm', '--SVM', type=int, help='using Logistics Regression', default=0, choices=[0, 1])
#
# p.add_argument('-knn', '--KNN', type=int, help='using Logistics Regression', default=0, choices=[0, 1])
p.add_argument('-k', '--K', type=int, help='value of k for KNN', default=5)

# p.add_argument('-dt', '--DT', type=int, help='using Logistics Regression', default=0, choices=[0, 1])
# p.add_argument('-nb', '--NB', type=int, help='using Logistics Regression', default=0, choices=[0, 1])
# p.add_argument('-bag', '--Bagging', type=int, help='using Logistics Regression', default=0, choices=[0, 1])
# p.add_argument('-rf', '--RF', type=int, help='using Logistics Regression', default=0, choices=[0, 1])
# p.add_argument('-ab', '--AB', type=int, help='using Logistics Regression', default=0, choices=[0, 1])
# p.add_argument('-gb', '--GB', type=int, help='using Logistics Regression', default=0, choices=[0, 1])
# p.add_argument('-lda', '--LDA', type=int, help='using Logistics Regression', default=0, choices=[0, 1])

p.add_argument('-m', '--model', type=str, help='choose a model', default='LR', choices=['LR','SVM','KNN','DT','SVM','NB','Bagging','RF','AB','GB','LDA',])

args = p.parse_args()

train(args)
