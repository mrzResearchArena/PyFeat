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

F = open('evaluationResults.txt', 'w')

F.write('Evaluation Scale:'+'\n')
F.write('0.0% <=Accuracy<= 100.0%'+'\n')
F.write('0.0 <=auROC<= 1.0'+'\n')
F.write('0.0 <=auPR<= 1.0'+'\n')  # average_Precision
F.write('0.0 <=F1_Score<= 1.0'+'\n')
F.write('-1.0 <=MCC<= 1.0'+'\n')
F.write('0.0%<=Sensitivity<= 100.0%'+'\n')
F.write('0.0%<=Specificity<= 100.0%'+'\n')

def runClassifiers(args):

    D = pd.read_csv(args.dataset, header=None)  # Using R
    # print('Before drop duplicates: {}'.format(D.shape))
    # D = D.drop_duplicates()  # Return : each row are unique value
    # print('After drop duplicates: {}\n'.format(D.shape))

    X = D.iloc[:, :-1].values
    y = D.iloc[:, -1].values

    from sklearn.preprocessing import StandardScaler
    scale = StandardScaler()
    X = scale.fit_transform(X)

    F.write('\n'+'------------------------------------------'+'\n')
    F.write('Using {} fold-cross validation results.\n'.format(args.nFCV))
    F.write('------------------------------------------'+'\n')

    Results = []  # compare algorithms

    from sklearn.metrics import accuracy_score, \
        log_loss, \
        classification_report, \
        confusion_matrix, \
        roc_auc_score,\
        average_precision_score,\
        auc,\
        roc_curve, f1_score, recall_score, matthews_corrcoef, auc

    # Step 05 : Spliting with 10-FCV :
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=args.nFCV, shuffle=True)

    for classifier, name in zip(Classifiers, Names):
        accuray = []
        auROC = []
        avePrecision = []
        F1_Score = []
        AUC = []
        MCC = []
        Recall = []
        LogLoss = []

        mean_TPR = 0.0
        mean_FPR = np.linspace(0, 1, 100)

        CM = np.array([
            [0, 0],
            [0, 0],
        ], dtype=int)

        print('{} is done.'.format(classifier.__class__.__name__))
        F.write(classifier.__class__.__name__+'\n\n')

        model = classifier
        for (train_index, test_index) in cv.split(X, y):

            X_train = X[train_index]
            X_test = X[test_index]

            y_train = y[train_index]
            y_test = y[test_index]

            # Step 06 : Scaling the feature
            # from sklearn.preprocessing import StandardScaler, MinMaxScaler
            #
            # scale = StandardScaler()
            # X_train = scale.fit_transform(X_train)
            # X_test = scale.transform(X_test)

            # model = BaggingClassifier(classifier)
            model.fit(X_train, y_train)

            # print(model.predict(X_train))

            # Calculate ROC Curve and Area the Curve
            y_proba = model.predict_proba(X_test)[:, 1]
            FPR, TPR, _ = roc_curve(y_test, y_proba)
            mean_TPR += np.interp(mean_FPR, FPR, TPR)
            mean_TPR[0] = 0.0
            roc_auc = auc(FPR, TPR)
            ##########################################
            # print(FPR)
            # print(TPR)
            ##########################################

            y_artificial = model.predict(X_test)

            auROC.append(roc_auc_score(y_true=y_test, y_score=y_proba))

            accuray.append(accuracy_score(y_pred=y_artificial, y_true=y_test))
            avePrecision.append(average_precision_score(y_true=y_test, y_score=y_proba)) # auPR
            F1_Score.append(f1_score(y_true=y_test, y_pred=y_artificial))
            MCC.append(matthews_corrcoef(y_true=y_test, y_pred=y_artificial))
            Recall.append(recall_score(y_true=y_test, y_pred=y_artificial))
            AUC.append(roc_auc)

            CM += confusion_matrix(y_pred=y_artificial, y_true=y_test)

        accuray = [_*100.0 for _ in accuray]
        Results.append(accuray)

        mean_TPR /= cv.get_n_splits(X, y)
        mean_TPR[-1] = 1.0
        mean_auc = auc(mean_FPR, mean_TPR)
        plt.plot(
            mean_FPR,
            mean_TPR,
            linestyle='-',
            label='{} ({:0.3f})'.format(name, mean_auc), lw=2.0)

        F.write('Accuracy: {0:.4f}%\n'.format(np.mean(accuray)))
        # print('auROC: {0:.6f}'.format(np.mean(auROC)))
        F.write('auROC: {0:.4f}\n'.format(mean_auc))
        # F.write('AUC: {0:.4f}\n'.format( np.mean(AUC)))
        F.write('auPR: {0:.4f}\n'.format(np.mean(avePrecision))) # average_Precision
        F.write('F1_Score: {0:.4f}\n'.format(np.mean(F1_Score)))
        F.write('MCC: {0:.4f}\n'.format(np.mean(MCC)))

        TN, FP, FN, TP = CM.ravel()
        F.write('Recall: {0:.4f}\n'.format( np.mean(Recall)) )
        F.write('Sensitivity: {0:.4f}%\n'.format( ((TP) / (TP + FN))*100.0 ))
        F.write('Specificity: {0:.4f}%\n'.format( ((TN) / (TN + FP))*100.0 ))
        F.write('Confusion Matrix:\n')
        F.write(str(CM)+'\n')
        F.write('_______________________________________'+'\n')

    ##########
    F.close()
    ##########

    ### auROC Curve ###
    if args.auROC == 1:
        auROCplot()
    ### boxplot algorithm comparison ###
    if args.boxPlot == 1:
        boxPlot(Results, Names)
    ### --- ###

    print('\nPlease, eyes on evaluationResults.txt')


def boxPlot(Results, Names):
    ### Algoritms Comparison ###
    # boxplot algorithm comparison
    fig = plt.figure()
    # fig.suptitle('Classifier Comparison')
    ax = fig.add_subplot(111)
    ax.yaxis.grid(True)
    plt.boxplot(Results, patch_artist=True, vert = True, whis=True, showbox=True)
    ax.set_xticklabels(Names, fontsize = 12)
    plt.xlabel('Classifiers', fontsize = 12,fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize = 12,fontweight='bold')

    plt.savefig('Accuracy_boxPlot.png', dpi=300)
    plt.show()
    ### --- ###

def auROCplot():
    ### auROC ###
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig('auROC.png', dpi=300)
    plt.show()
    ### --- ###


if __name__ == '__main__':
    # print('Please, enter number of cross validation:')
    import argparse
    p = argparse.ArgumentParser(description='Run Machine Learning Classifiers.')

    p.add_argument('-cv', '--nFCV', type=int, help='Number of crossValidation', default=10)
    p.add_argument('-data', '--dataset', type=str, help='~/dataset.csv', default='optimumDataset.csv')
    p.add_argument('-roc', '--auROC', type=int, help='Print ROC Curve', default=1, choices=[0, 1])
    p.add_argument('-box', '--boxPlot', type=int, help='Print Accuracy Box Plaot', default=1, choices=[0, 1])

    args = p.parse_args()

    runClassifiers(args)



