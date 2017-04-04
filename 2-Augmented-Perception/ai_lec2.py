import sklearn
import sklearn.feature_extraction.text
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.datasets
import sklearn.cluster
import sklearn.decomposition
import sklearn.metrics
import sklearn.metrics.pairwise
import sklearn.manifold

import scipy #For hierarchical clustering and some visuals
#import scipy.cluster.hierarchy
import gensim#For topic modeling
import nltk #the Natural Language Toolkit
import requests #For downloading our datasets
import numpy as np #for arrays
import pandas #gives us DataFrames
import matplotlib.pyplot as plt #For graphics
import matplotlib.cm #Still for graphics
import seaborn #Makes the graphics look nicer

import json #For reading data from github's API
import os #For looking through files
import os.path #For managing file paths

def loadDir(targetDir, category):
    allFileNames = os.listdir(targetDir)
    #We need to make them into useable paths and filter out hidden files
    filePaths = [os.path.join(targetDir, fname) for fname in allFileNames if fname[0] != '.']

    #The dict that will become the DataFrame
    senDict = {
        'category' : [category] * len(filePaths),
        'filePath' : [],
        'text' : [],
    }

    for fPath in filePaths:
        with open(fPath) as f:
            senDict['text'].append(f.read())
            senDict['filePath'].append(fPath)

    return pandas.DataFrame(senDict)

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

def loadSenateSmall():
    print("Loading senate data")
    senReleasesDF = pandas.read_csv("data/senReleasesDF.csv")
    senReleasesDF = senReleasesDF.dropna(axis=0, how='any')

    print("Converting to vectors")
    senReleasesDF['tokenized_sents'] = senReleasesDF['text'].apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
    senReleasesDF['normalized_sents'] = senReleasesDF['tokenized_sents'].apply(lambda x: [normlizeTokens(s, stopwordLst = stop_words_nltk, stemmer = None) for s in x])

    senReleasesDF['tokenized_text'] = senReleasesDF['text'].apply(lambda x: nltk.word_tokenize(x))
    senReleasesDF['normalized_text'] = senReleasesDF['tokenized_text'].apply(lambda x: normlizeTokens(x))

    ngCountVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5, min_df=3, stop_words='english', norm='l2')
    newsgroupsVects = ngCountVectorizer.fit_transform([' '.join(l) for l in senReleasesDF['normalized_text']])
    senReleasesDF['vect'] = [np.array(v) for v in newsgroupsVects.todense()]

    senReleasesDF['category'] = senReleasesDF['targetSenator']

    return senReleasesDF


def loadSenateLarge():
    dataDir = 'data/grimmerPressReleases'
    senReleasesDF = pandas.DataFrame()

    for senatorName in [d for d in os.listdir(dataDir) if d[0] != '.']:
        print("Loading senator: {}".format(senatorName))
        senPath = os.path.join(dataDir, senatorName)
        senReleasesDF = senReleasesDF.append(loadDir(senPath, senatorName), ignore_index = True)

    print("Converting to vectors")
    stop_words_nltk = nltk.corpus.stopwords.words('english')
    snowball = nltk.stem.snowball.SnowballStemmer('english')
    wordnet = nltk.stem.WordNetLemmatizer()

    senReleasesDF['tokenized_sents'] = senReleasesDF['text'].apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
    senReleasesDF['normalized_sents'] = senReleasesDF['tokenized_sents'].apply(lambda x: [normlizeTokens(s, stopwordLst = stop_words_nltk, stemmer = None) for s in x])

    senReleasesDF['tokenized_text'] = senReleasesDF['text'].apply(lambda x: nltk.word_tokenize(x))
    senReleasesDF['normalized_text'] = senReleasesDF['tokenized_text'].apply(lambda x: normlizeTokens(x))

    ngCountVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5, min_df=3, stop_words='english', norm='l2')
    newsgroupsVects = ngCountVectorizer.fit_transform([' '.join(l) for l in senReleasesDF['normalized_text']])
    senReleasesDF['vect'] = [np.array(v) for v in newsgroupsVects.todense()]

    senReleasesDF['targetSenator'] = senReleasesDF['category']
    return senReleasesDF

def loadNatregimes(categoryVar = 'HC90', numClusters = 7):
    print("loading data")
    natregimesDF = pandas.read_csv('data/natregimes.csv')

    print("Converting to vectors")

    nonData = ['REGIONS',
     'NOSOUTH',
     'POLY_ID',
     'NAME',
     'STATE_NAME',
     'STATE_FIPS',
     'CNTY_FIPS',
     'FIPS',
     'STFIPS',
     'COFIPS',
     'FIPSNO',
     'SOUTH',
     'West']
    natregimesDataDF = natregimesDF.drop(nonData, axis=1).astype('float')
    natregimesDF['vect'] = [v.reshape(1, -1) for v in natregimesDataDF.as_matrix()]

    populationSeries = natregimesDF[categoryVar].astype('float')

    quartiles = []
    for i in range(numClusters - 1):
        quartiles.append(np.percentile(populationSeries, (1 + i) / numClusters * 100))

    def categorizer(popNum):
        for i, cutVal in enumerate(quartiles):
            if popNum <= cutVal:
                return 'Q{}'.format(i + 1)
        #floating point numbers are tricky
        return 'Q{}'.format(len(quartiles) + 1)

    natregimesDF['category'] = [categorizer(p) for p in populationSeries]
    return natregimesDF

def loadNYTmodel():
    return gensim.models.word2vec.Word2Vec.load_word2vec_format('data/nytimes_cbow.reduced.txt')

def clusteringMetrics(clf, df):
    print("Homogeneity: {:0.3f}".format(sklearn.metrics.homogeneity_score(df['category'], clf.labels_)))
    print("Completeness: {:0.3f}".format(sklearn.metrics.completeness_score(df['category'], clf.labels_)))
    print("V-measure: {:0.3f}".format(sklearn.metrics.v_measure_score(df['category'], clf.labels_)))

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

# Word2Vec
def makeWord2Vec(df):
    return gensim.models.word2vec.Word2Vec(df['normalized_sents'].sum())

def plotWord2Vec(w2v, numWords = 50):
    targetWords = w2v.index2word[:numWords]
    wordsSubMatrix = []
    for word in targetWords:
        wordsSubMatrix.append(w2v[word])
    wordsSubMatrix = np.array(wordsSubMatrix)

    pcaWords = sklearn.decomposition.PCA(n_components = 50).fit(wordsSubMatrix)
    reducedPCA_data = pcaWords.transform(wordsSubMatrix)
    #T-SNE is theoretically better, but you should experiment
    tsneWords = sklearn.manifold.TSNE(n_components = 2).fit_transform(reducedPCA_data)

    fig = plt.figure(figsize = (10,6))
    ax = fig.add_subplot(111)
    ax.set_frame_on(False)
    plt.scatter(tsneWords[:, 0], tsneWords[:, 1], alpha = 0)#Making the points invisible
    for i, word in enumerate(targetWords):
        ax.annotate(word, (tsneWords[:, 0][i],tsneWords[:, 1][i]), size =  20 * (numWords - i) / numWords)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    plt.close()
