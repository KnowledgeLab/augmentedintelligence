#All these packages need to be installed from pip
#For ML
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

#Displays the graphs
import graphviz #You also need to install the command line graphviz

#For NLP
import nltk
from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger
from nltk.parse import stanford
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
from nltk.draw.tree import TreeView
from nltk.tokenize import sent_tokenize


import numpy as np #arrays
import matplotlib.pyplot as plt #Plots
from matplotlib.colors import ListedColormap
import seaborn #Makes plots look nice, also heatmaps
import scipy as sp #for interp

#These are from the standard library
import collections
import os
import os.path
import random
import re
import glob
import pandas
import requests
import json
import math
import tarfile
import zipfile
import io


def normlizeTokens(tokenLst, stopwordLst = None, stemmer = None, lemmer = None, vocab = None):
    #We can use a generator here as we just need to iterate over it

    #Lowering the case and removing non-words
    workingIter = (w.lower() for w in tokenLst if w.isalpha())

    #Now we can use the semmer, if provided
    if stemmer is not None:
        workingIter = (stemmer.stem(w) for w in workingIter)

    #And the lemmer
    if lemmer is not None:
        workingIter = (lemmer.lemmatize(w) for w in workingIter)

    #And remove the stopwords
    if stopwordLst is not None:
        workingIter = (w for w in workingIter if w not in stopwordLst)

    #We will return a list with the stopwords removed
    if vocab is not None:
        vocab_str = '|'.join(vocab)
        workingIter = (w for w in workingIter if re.match(vocab_str, w))

    return list(workingIter)

#Cluster Detection

def loadNewsGroups(categories = ['comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos']):
    newsgroupsCategories = categories
    newsgroups = sklearn.datasets.fetch_20newsgroups(subset='train', data_home = 'data/sklearnData')
    newsgroupsDF = pandas.DataFrame(columns = ['text', 'category', 'source_file'])

    for category in newsgroupsCategories:
        print("Loading data for: {}".format(category))
        ng = sklearn.datasets.fetch_20newsgroups(subset='train', categories = [category], remove=['headers', 'footers', 'quotes'], data_home = 'data')
        newsgroupsDF = newsgroupsDF.append(pandas.DataFrame({'text' : ng.data, 'category' : [category] * len(ng.data), 'source_file' : ng.filenames}), ignore_index=True)

    print("Converting to vectors")
    #tokenize
    newsgroupsDF['tokenized_text'] = newsgroupsDF['text'].apply(lambda x: nltk.word_tokenize(x))
    newsgroupsDF['normalized_text'] = newsgroupsDF['tokenized_text'].apply(lambda x: normlizeTokens(x))

    ngCountVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5, min_df=3, stop_words='english', norm='l2')
    newsgroupsVects = ngCountVectorizer.fit_transform([' '.join(l) for l in newsgroupsDF['normalized_text']])
    newsgroupsDF['vect'] = [np.array(v) for v in newsgroupsVects.todense()]

    return newsgroupsDF

def loadEmailZip(targetFile, category):
    # regex for stripping out the leading "Subject:" and any spaces after it
    subject_regex = re.compile(r"^Subject:\s+")

    #The dict that will become the DataFrame
    emailDict = {
        'category' : [],
        'text' : [],
    }
    with tarfile.open(targetFile) as tar:
        for tarinfo in tar.getmembers():
            if tarinfo.isreg():
                with tar.extractfile(tarinfo) as f:
                    s = f.read().decode('latin1', 'surrogateescape')
                    for line in s.split('\n'):
                        if line.startswith("Subject:"):
                            #Could also save the subject field
                            subject = subject_regex.sub("", line).strip()
                            emailDict['text'].append(subject)
    emailDict['category'] = [category] * len(emailDict['text'])
    return pandas.DataFrame(emailDict)

def loadSpam(holdBackFraction = .2):
    spamDF = loadEmailZip('data/Spam_Data/20021010_spam.tar.bz2', 'spam')
    spamDF = spamDF.append(loadEmailZip('data/Spam_Data/20021010_hard_ham.tar.bz2', 'not spam'), ignore_index= True)
    spamDF = spamDF.append(loadEmailZip('data/Spam_Data/20021010_easy_ham.tar.bz2', 'not spam'), ignore_index= True)
    spamDF['is_spam'] = [c == 'spam' for c in spamDF['category']]

    stop_words_nltk = nltk.corpus.stopwords.words('english')
    snowball = nltk.stem.snowball.SnowballStemmer('english')


    spamDF['tokenized_emails'] = spamDF['text'].apply(lambda x: nltk.word_tokenize(x))
    spamDF['normalized_emails'] = spamDF['tokenized_emails'].apply(lambda x: normlizeTokens(x, stopwordLst = None, stemmer = None))

    shuffledSpamDF = spamDF.reindex(np.random.permutation(spamDF.index))
    holdBackIndex = int(holdBackFraction * len(shuffledSpamDF))
    train_data = shuffledSpamDF[holdBackIndex:].copy()
    test_data = shuffledSpamDF[:holdBackIndex].copy()

    return train_data, test_data

def count_words(traingDF, textColumn, trainingColumn):
    counts = collections.defaultdict(lambda: [0, 0])
    for index, row in traingDF.iterrows():
        for word in set(row[textColumn]):
            if row[trainingColumn]:
                counts[word][0] += 1
            else:
                counts[word][1] += 1
    return counts

def word_probabilities(counts, total_spams, total_non_spams, k=0.5):
    """turn the word_counts into a list of triplets
    w, p(w | spam) and p(w | ~spam)"""
    retTuples = []
    for w, (spam, non_spam) in counts.items():
        retTuples.append((w,
             (spam + k) / (total_spams + 2 * k),
             (non_spam + k) / (total_non_spams + 2 * k)))
    return retTuples

def spam_probability(word_probs, message_words):
    #message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0 #Initialize; we are working with log probs to deal with underflow.

    for word, prob_if_spam, prob_if_not_spam in word_probs: #We iterate over all possible words we've observed
        # for each word in the message,
        # add the log probability of seeing it
        if word in message_words:
            log_prob_if_spam += math.log(prob_if_spam) #This is prob of seeing word if spam
            log_prob_if_not_spam += math.log(prob_if_not_spam) #This is prob of seeing word if not spam

        # for each word that's not in the message
        # add the log probability of _not_ seeing it
        else:
            log_prob_if_spam += math.log(1.0 - prob_if_spam)
            log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)
    P = 1/(1 + math.exp(log_prob_if_not_spam - log_prob_if_spam))
    #prob_if_spam = math.exp(log_prob_if_spam) #Compute numerator
    #prob_if_not_spam = math.exp(log_prob_if_not_spam)
    #return prob_if_spam / (prob_if_spam + prob_if_not_spam) #Compute whole thing and return
    return P

class NaiveBayesClassifier:

    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = [] #Initializes word_probs as an empty list, sets a default smoothing parameters

    def train(self, training_set, trainingColumn, textColumn): #Operates on the training_set

        # count spam and non-spam messages: first step of training
        num_spams = training_set[trainingColumn].value_counts()[True]
        num_non_spams = len(training_set) - num_spams

        # run training data through our "pipeline"
        word_counts = count_words(training_set, textColumn, trainingColumn)
        self.word_probs = word_probabilities(word_counts,
                                             num_spams,
                                             num_non_spams,
                                             self.k) #"Train" classifier

    def classify(self, message):
        return spam_probability(self.word_probs, message) #Now we have all we need to classify a message

def evaluateClassifier(clf, testDF):
    classified = [(row['category'], clf.classify(row['normalized_emails']))
                  for index, row in testDF.iterrows()]
    counts = collections.Counter((actual == 'spam', predicted_probability > 0.5)
                         for actual, predicted_probability in classified)
    precision = counts[(True,True)]/(counts[(False,True)]+counts[(True,True)]) #True positives over all positive predictions
    print(precision)

    recall = counts[(True,True)]/(counts[(True,False)]+counts[(True,True)])#what fraction of positives identified
    print(recall)

    f_measure = 2 * (precision * recall)/(precision + recall)
    print (f_measure)

def plotClassifier(clf, testDF):
    x, y, _ = sklearn.metrics.roc_curve(testDF['is_spam'], [clf.classify(d) for d in testDF['normalized_emails']])
    roc_auc = sklearn.metrics.auc(x,y)
    plt.figure()
    plt.plot(x,y, color = 'darkorange', lw = 2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()
    plt.close()
