import numpy as np #arrays
import matplotlib.pyplot as plt #Plots

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sklearn
import sklearn.feature_extraction.text
import sklearn.decomposition
from sklearn import preprocessing, linear_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.datasets import fetch_20newsgroups, make_blobs
from sklearn.feature_extraction.text import TfidfVectorizer  #Feature extraction
from sklearn.naive_bayes import MultinomialNB #Our learner.
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors

import numpy as np #arrays
import matplotlib.pyplot as plt #Plots
from matplotlib.colors import ListedColormap
import seaborn #Makes plots look nice, also heatmaps
import scipy as sp #for interp
import pandas

def evaluateClassifier(clf, testDF):
    predictions = clf.predict(np.stack(testDF['vect'], axis=0))
    classes = []
    results = {
        'Error_Rate' : [],
        'AUC' : [],
        'Precision' : [],
        'Average_Precision' : [],
        'Recall' : [],
        }

    for cat in set(testDF['category']):
        preds = [True if (c == cat) else False for c in predictions]
        acts = [True if (c == cat) else False for c in testDF['category']]
        classes.append(cat)
        results['AUC'].append(sklearn.metrics.roc_auc_score(preds, acts))
        results['Average_Precision'].append(sklearn.metrics.average_precision_score(preds, acts))
        results['Precision'].append(sklearn.metrics.precision_score(preds, acts))
        results['Recall'].append(sklearn.metrics.recall_score(preds, acts))
        results['Error_Rate'].append(1 -  sklearn.metrics.accuracy_score(preds, acts))
    df = pandas.DataFrame(results, index=classes)
    df.index.rename('Category', inplace=True)
    return df


def plotMultiROC(clf, testDF):
    #By making the column names variables we can easily use this function on new data sets

    #Get the names of each of the possible classes and the probabiltiess
    classes = clf.classes_
    try:
        probs = clf.predict_proba(np.stack(testDF['vect'], axis=0))
    except AttributeError:
        print("The {} classifier does not apear to support prediction probabilties, so an ROC curve can't be created. You can try adding `probability = True` to the model specification or use a different model.".format(type(clf)))
        return
    predictions = clf.predict(np.stack(testDF['vect'], axis=0))

    #setup axis for plotting
    fig, ax = plt.subplots()

    #We can return the AUC values, in case they are useful
    aucVals = []
    for classIndex, className in enumerate(classes):        #Setup binary classes
        truths = [1 if c == className else 0 for c in testDF['category']]
        predict = [1 if c == className else 0 for c in predictions]
        scores = probs[:, classIndex]

        #Get the ROC curve
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(truths, scores)
        auc = sklearn.metrics.auc(fpr, tpr)
        aucVals.append(auc)

        #Plot the class's line
        ax.plot(fpr, tpr, label = "{} (AUC ${:.3f}$)".format(str(className).split(':')[0], auc))

    #Make the plot nice, then display it
    ax.set_title('Receiver Operating Characteristics')
    plt.plot([0,1], [0,1], color = 'k', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc = 'lower right')
    plt.show()
    plt.close()
    #return aucVals

def plotConfusionMatrix(clf, testDF):
    predictions = clf.predict(np.stack(testDF['vect'], axis=0))
    mat = confusion_matrix(predictions, testDF['category'])
    seaborn.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels=testDF['category'].unique(), yticklabels=testDF['category'].unique())
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.title("Confusion Matrix")
    plt.show()
    plt.close()


def visulizeClusters(clf, df):
    pca = sklearn.decomposition.PCA(n_components = 2).fit(np.stack(df['vect'], axis=1)[0])
    reduced_data = pca.transform(np.stack(df['vect'], axis=1)[0])

    categories = list(set(df['category']))
    palletTrue = seaborn.color_palette("Paired", len(categories))
    palletPred = seaborn.color_palette("cubehelix", len(set((clf.labels_))))

    coloursDict = {c : palletTrue[i] for i, c in enumerate(categories)}
    coloursTrue = [coloursDict[c] for c in df['category']]
    colorsPred = [palletPred[l] for l in clf.labels_]


    fig = plt.figure(figsize = (8,12))
    axT = fig.add_subplot(211)
    axP = fig.add_subplot(212)

    axT.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha = 0.5, color = coloursTrue)
    axT.set_title('True Classes')

    axP.scatter(reduced_data[:, 0], reduced_data[:, 1], color = colorsPred, alpha = 0.5)
    axP.set_title('Predicted Clusters')
    plt.show()
    plt.close()
