### Avoid warning ###
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn


### Essential ###
import pandas as pd
import numpy as np


### Fixed State ###
# np.random.seed(seed=111)

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


    # ### Scaling using hands-on ###
    # storeMeanSD = []
    # for i in range(X_train.shape[1]):
    #     eachFeature = X_train[:, i]
    #     MEAN = np.mean(eachFeature)
    #     SD = np.std(eachFeature)
    #     ### Stored mean and standard deviation for each feature ###
    #     storeMeanSD.append((MEAN, SD))
    #
    # storeMeanSD = np.array(storeMeanSD)
    #
    #
    # scalingX_train = []
    # for i in range(X_train.shape[1]):
    #     eachFeature = X_train[:, i]
    #     v = (eachFeature - storeMeanSD[i][0]) / (storeMeanSD[i][1])
    #     scalingX_train.append(v)
    #
    # X_train = np.array(scalingX_train).T
    #


    ### Run Machine Learning Model (best)  ###
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2', C=0.10, max_iter=500, solver='sag')


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

    print('Model training is done.')


import argparse

p = argparse.ArgumentParser(description='Training the model with LR.')
p.add_argument('-data', '--dataset', type=str, help='~/dataset.csv', default='optimumDataset.csv')
args = p.parse_args()

train(args)

