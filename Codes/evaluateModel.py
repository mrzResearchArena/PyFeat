import numpy as np
import pandas as pd

def evaluate(args):

    ### Test Dataset ###
    D = pd.read_csv(args.testDatasetPath, header=None)
    X = D.iloc[:, :-1].values
    y_test = D.iloc[:, -1].values

    F = open('selectedIndex.txt', 'r')
    v = F.read().split(',')
    v = [int(i) for i in v]
    X_test = X[:, v]

    ### --- ###


    ### Hands-on scaling on train dataset  ###

    D = pd.read_csv(args.optimumDatasetPath, header=None)
    X_train = D.iloc[:, :-1].values
    y_train = D.iloc[:, -1].values

    storeMeanSD = []
    for i in range(X_train.shape[1]):
        eachFeature = X_train[:, i]
        MEAN = np.mean(eachFeature)
        SD = np.std(eachFeature)
        ### Stored mean and standard deviation for each feature ###
        storeMeanSD.append((MEAN, SD))

    storeMeanSD = np.array(storeMeanSD)


    scalingX_test = []
    for i in range(X_test.shape[1]):
        eachFeature = X_test[:, i]
        v = (eachFeature - storeMeanSD[i][0]) / (storeMeanSD[i][1])
        scalingX_test.append(v)

    ### --- ###

    ### Scaling X_test using X_train ###
    X_test = np.array(scalingX_test).T
    ### --- ###


    import ensure
    ensure.testModel(X_test, y_test)

import argparse
p = argparse.ArgumentParser(description='Training the model with LR.')
p.add_argument('-optimumPath', '--optimumDatasetPath', type=str, help='~/dataset.csv', default='optimumDataset.csv')
p.add_argument('-testPath', '--testDatasetPath', type=str, help='~/dataset.csv', default='testDataset.csv')
args = p.parse_args()

evaluate(args)


