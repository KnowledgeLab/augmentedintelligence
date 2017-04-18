#All these packages need to be installed from pip

import nltk #For POS tagging
import sklearn #For generating some matrices
import pandas #For DataFrames
import numpy as np #For arrays
import matplotlib.pyplot as plt #For plotting
import seaborn #MAkes the plots look nice
import IPython.display #For displaying images

#This 'magic' command makes the plots work better
#in the notebook, don't use it outside of a notebook.
#Also you can ignore the warning
#%matplotlib inline

#This has some C components and installs are OS specific
#See http://igraph.org/python/ for details
import igraph as ig #For the networks


import pickle #if you want to save layouts

stop_words_nltk = nltk.corpus.stopwords.words('english')
snowball = nltk.stem.snowball.SnowballStemmer('english')

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

def loadNewsGroups(num = 100, categories = ('comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos')):
    newsgroupsCategories = categories
    newsgroups = sklearn.datasets.fetch_20newsgroups(subset='train', data_home = 'data')
    newsgroupsDF = pandas.DataFrame(columns = ['text', 'category', 'source_file'])

    for category in newsgroupsCategories:
        print("Loading data for: {}".format(category))
        ng = sklearn.datasets.fetch_20newsgroups(subset='train', categories = [category], remove=['headers', 'footers', 'quotes'], data_home = 'data')
        newsgroupsDF = newsgroupsDF.append(pandas.DataFrame({'text' : ng.data, 'category' : [category] * len(ng.data), 'source_file' : ng.filenames})[:num // len(categories) + 1], ignore_index=True)

    newsgroupsDF['sentences'] = newsgroupsDF['text'].apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
    #newsgroupsDF.index = range(len(newsgroupsDF) - 1, -1,-1) #Reindex to make things nice in the future
    newsgroupsDF['normalized_sents'] = newsgroupsDF['sentences'].apply(lambda x: [normlizeTokens(s, stopwordLst = None, stemmer = snowball) for s in x])
    return newsgroupsDF

def loadReddit(num = 100):
    print("Loading Reddit data")
    redditDF = pandas.read_csv('data/reddit.csv', index_col = 0)

    redditTopScores = redditDF.sort_values('score')[-num:]
    redditTopScores['sentences'] = redditTopScores['text'].apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
    redditTopScores.index = range(len(redditTopScores) - 1, -1,-1) #Reindex to make things nice in the future
    redditTopScores['normalized_sents'] = redditTopScores['sentences'].apply(lambda x: [normlizeTokens(s, stopwordLst = None, stemmer = snowball) for s in x])

    return redditTopScores


def loadSenate(num = 100):
    print("Loading senate data")
    senReleasesDF = pandas.read_csv("data/senReleasesTraining.csv")
    senReleasesDF = senReleasesDF.dropna(axis=0, how='any')[:num]

    senReleasesDF['sentences'] = senReleasesDF['text'].apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
    senReleasesDF['normalized_sents'] = senReleasesDF['sentences'].apply(lambda x: [normlizeTokens(s, stopwordLst = None, stemmer = snowball) for s in x])
    return senReleasesDF

def wordCooccurrence(sentences, makeMatrix = False):
    words = set()
    for sent in sentences:
        words |= set(sent)
    wordLst = list(words)
    wordIndices = {w: i for i, w in enumerate(wordLst)}
    wordCoCounts = {}
    #consider a sparse matrix if memory becomes an issue
    coOcMat = np.zeros((len(wordIndices), len(wordIndices)))
    for sent in sentences:
        for i, word1 in enumerate(sent):
            word1Index = wordIndices[word1]
            for word2 in sent[i + 1:]:
                coOcMat[word1Index][wordIndices[word2]] += 1
    if makeMatrix:
        return coOcMat, wordLst
    else:
        coOcMat = coOcMat.T + coOcMat
        edges = list(zip(*np.where(coOcMat)))
        weights = coOcMat[np.where(coOcMat)]
        g = ig.Graph( n = len(wordLst),
            edges = edges,
            vertex_attrs = {'name' : wordLst, 'label' : wordLst},
            edge_attrs = {'weight' : weights}
                    )
        return g

def wordCooccurrence(sentences, makeMatrix = False):
    words = set()
    for sent in sentences:
        words |= set(sent)
    wordLst = list(words)
    wordIndices = {w: i for i, w in enumerate(wordLst)}
    wordCoCounts = {}
    #consider a sparse matrix if memory becomes an issue
    coOcMat = np.zeros((len(wordIndices), len(wordIndices)))
    for sent in sentences:
        for i, word1 in enumerate(sent):
            word1Index = wordIndices[word1]
            for word2 in sent[i + 1:]:
                coOcMat[word1Index][wordIndices[word2]] += 1
    if makeMatrix:
        return coOcMat, wordLst
    else:
        coOcMat = coOcMat.T + coOcMat
        edges = list(zip(*np.where(coOcMat)))
        weights = coOcMat[np.where(coOcMat)]
        g = ig.Graph( n = len(wordLst),
            edges = edges,
            vertex_attrs = {'name' : wordLst, 'label' : wordLst},
            edge_attrs = {'weight' : weights}
                    )
        return g

def posCooccurrence(sentences, *posType, makeMatrix = False):
    pal = ig.RainbowPalette(n = len(posType))
    palMap = {p : pal.get(i) for i, p in enumerate(posType)}
    words = set()
    reducedSents = []
    #Only using the first kind of POS for each word
    wordsMap = {}
    for sent in sentences:
        s = [(w, t) for w, t in nltk.pos_tag(sent) if t in posType]
        for w, t in s:
            if w not in wordsMap:
                wordsMap[w] = t
        reducedSent = [w for w, t in s]
        words |= set(reducedSent)
        reducedSents.append(reducedSent)
    wordLst = list(words)
    wordIndices = {w: i for i, w in enumerate(wordLst)}
    wordCoCounts = {}
    #consider a sparse matrix if memory becomes an issue
    coOcMat = np.zeros((len(wordIndices), len(wordIndices)))
    for sent in reducedSents:
        for i, word1 in enumerate(sent):
            word1Index = wordIndices[word1]
            for word2 in sent[i + 1:]:
                coOcMat[word1Index][wordIndices[word2]] += 1
    if makeMatrix:
        return coOcMat, wordLst
    else:
        coOcMat = coOcMat.T + coOcMat
        edges = list(zip(*np.where(coOcMat)))
        weights = coOcMat[np.where(coOcMat)]
        kinds = [wordsMap[w] for w in wordLst]
        colours = [palMap[k] for k in kinds]
        g = ig.Graph( n = len(wordLst),
            edges = edges,
            vertex_attrs = {'name' : wordLst,
                            'label' : wordLst,
                            'kind' : kinds,
                            'color' : colours,
                            'label_color' : colours,
                           },
            edge_attrs = {'weight' : weights})
        return g

def reduceGraph(g, maxEdge = 1000, maxNode = 200):
    numEdge = g.ecount()
    numNode = g.vcount()
    flipper = True
    while numNode > maxNode:
        medianDegree = np.percentile(g.vs.degree(), 20)
        g = g.subgraph(g.vs.select(lambda x: x.degree() > medianDegree))
        numNode = g.vcount()
    while numEdge > maxEdge:
        medianEdge = np.percentile(g.es['weight'], 20)
        g = g.subgraph_edges(g.es.select(lambda x: x['weight'] > medianEdge))
        numEdge = g.ecount()
    return g

def displayBetweeness(g):
    val = np.array(g.betweenness())
    return _display(g, val)

def displayDegree(g):
    val = np.array(g.degree())
    return _display(g, val)

def displayEigen(g):
    val = np.array(g.eigenvector_centrality())
    return _display(g, val)

def displayCloseness(g):
    val = np.array(g.closeness())
    return _display(g, val)

def _display(g, val):
    gPlot = g.copy()
    layout = gPlot.layout_fruchterman_reingold()
    ranks = np.empty(len(val), int)
    ranks[val.argsort()] = np.arange(len(val))
    pal = ig.GradientPalette("red", "blue", gPlot.vcount())
    gPlot.vs['label_color'] = [pal[int(v)] for v in ranks]
    gPlot.vs['color'] = [pal[int(v)] for v in ranks]
    gPlot.vs['label_size'] = (np.abs(30 + 40 * (val / np.max(val) - 1))).tolist()
    return ig.plot(gPlot, layout = layout, edge_width = 0.2, vertex_size = .5)

def displayNeighbours(g, nodeName, step = 2):
    try:
        neighbors = set(g.neighbors(nodeName))
    except ValueError:
        raise ValueError("The node {} is not in the network, maybe you filtered it out".format(nodeName))
    for i in range(step - 1):
        newCollection = set()
        for n in neighbors:
            newCollection |= set(g.neighbors(n))
        neighbors |= newCollection
    gPlot = g.subgraph(g.vs.select(neighbors))
    gPlot.vs.find(nodeName)['color'] = 'yellow'
    gPlot.vs.find(nodeName)['label_color'] = 'yellow'
    layout = gPlot.layout_fruchterman_reingold()
    ig.plot(gPlot, layout = layout, edge_width = 0.2, target = 'data/temp.png', vertex_size = .5, label_size = 40)
    return 'data/temp.png'

def displayGraph(g):
    layout = g.layout_fruchterman_reingold(weights = 'weight', repulserad = 10)
    ig.plot(g, layout = layout, edge_width = 0.2, target = 'data/temp.png', vertex_size = .5, label_size = 40)
    return 'data/temp.png'
