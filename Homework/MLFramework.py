import numpy as np
import itertools
from Feature import Feature
from sklearn import tree
import pydotplus

def loadData(inputItr, features, label=None, rowLimit=None):
    if rowLimit:
        inputItr = itertools.islice(inputItr, rowLimit)
    else:
        inputItr = inputItr

    rows = list(inputItr)
    numRows = len(rows)
    data = np.empty((numRows, len(features)), dtype=np.float_)
    if label is not None:
        labelData = np.empty((len(rows),), dtype=np.float_)

    for i, row in enumerate(rows):
        if not row:
            continue
        for j, feature in enumerate(features):
            data[i][j] = feature.extractFeature(row)

        if label is not None:
            labelData[i] = label.extractFeature(row)

    if label is not None:
        return data, labelData
    else:
        return data

def crossValidate(models, data, labels, k):
    partitions = np.arange(data.shape[0]) % k
    np.random.shuffle(partitions)
    masks = dict()
    for partition in xrange(k):
        smallMask = (partitions == partition).nonzero()
        bigMask = (partitions != partition).nonzero()
        masks[partition] = (smallMask, bigMask)

    maxFitness, result = 0, None
    avFitnesses = np.empty((len(models),), dtype=np.float_)
    for i, model in enumerate(models):
        fitnesses = np.empty((k,), dtype=np.float_)
        for partition in xrange(k):
            smallMask, bigMask = masks[partition]

            trainingData = data[bigMask]
            testData = data[smallMask]
            trainingLabels = labels[bigMask]
            testLabels = labels[smallMask]

            model.fit(trainingData, trainingLabels)
            fitness = model.testModel(testData, testLabels)
            fitnesses[partition] = fitness
        avFitness = np.mean(fitnesses)
        avFitnesses[i] = avFitness
        # print "%s: %.4f" % (model, avFitness)
        if avFitness > maxFitness:
            maxFitness = avFitness
            result = model
    fitnessMean = np.mean(avFitnesses)
    fitnessStd = np.std(avFitnesses)
    # print fitnessMean, fitnessStd, maxFitness
    normalizedFitness = (maxFitness - fitnessMean) / fitnessStd

    return result, normalizedFitness

class SupervisedML(object):
    def __init__(self, features, label, *args, **kwargs):
        self.features = features
        self.label = label

        self.model = self.makeModel(*args, **kwargs)
        if args:
            if kwargs:
                self.setupArgs = (args, kwargs)
            else:
                self.setupArgs = args
        elif kwargs:
            self.setupArgs = kwargs
        else:
            self.setupArgs = None

    @classmethod
    def makeModels(cls, paramRanges, *args, **kwargs):
        models = list()

        params = paramRanges.keys()
        ranges = [paramRanges[param] for param in params]

        for paramValues in itertools.product(*ranges):
            paramsDict = dict([(param, v) for param, v in zip(params, paramValues)])
            kwargs.update(paramsDict)
            models.append(cls(*args, **kwargs))

        return models

    @classmethod
    def makeModel(self, *args, **kwargs):
        raise NotImplementedError

    def runModel(self, data):
        raise NotImplementedError

    def __str__(self):
        if self.setupArgs:
            return "%s(%s)" % (self.__class__.__name__, str(self.setupArgs))
        else:
            return "%s()" % self.__class__.__name__

class Classifier(SupervisedML):
    def loadTrainingData(self, inputItr, rowLimit=None):
        return self.loadData(inputItr, self.features, self.label, rowLimit)

    def loadTestData(self, inputItr, rowLimit=None):
        return self.loadData(inputItr, self.features, self.label, rowLimit)

    def fit(self, data, labels):
        raise NotImplementedError

    def testModel(self, data, labels):
        predictions = self.runModel(data)

        matched = [p == l for p, l in zip(predictions, labels)].count(True)
        return float(matched) / data.shape[0]

class DecisionTree(Classifier):
    def __init__(self, features, label, *args, **kwargs):
        super(DecisionTree, self).__init__(features, label, *args, **kwargs)

    def exportGraph(self, dest, labelNames=None):
        dot_data = tree.export_graphviz(self.model, out_file=None,
                        feature_names=[f.name for f in self.features],
                        class_names=labelNames, rounded=True, filled=True) 
        graph = pydotplus.graph_from_dot_data(dot_data) 
        graph.write_pdf(dest)

    def makeModel(self, *args, **kwargs):
        return tree.DecisionTreeClassifier(*args, **kwargs)

    def fit(self, data, labels):
        self.model.fit(data, labels)

    def runModel(self, data):
        return self.model.predict(data)