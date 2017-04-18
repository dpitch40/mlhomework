import numpy as np
import itertools
from Feature import Feature
from sklearn import tree
import pydotplus

class SupervisedML(object):
    def __init__(self, features, label, *args, **kwargs):
        self.features = features
        self.label = label

        self.setupModel(*args, **kwargs)

    def loadTrainingData(self, inputItr, rowLimit=None):
        return self.loadData(inputItr, True, rowLimit)

    def loadTestData(self, inputItr, rowLimit=None):
        return self.loadData(inputItr, False, rowLimit)

    def loadData(self, inputItr, withLabels, rowLimit):
        if rowLimit:
            self.inputItr = itertools.islice(inputItr, rowLimit)
        else:
            self.inputItr = inputItr

        rows = list(self.inputItr)
        numRows = len(rows)
        data = np.empty((numRows, len(self.features)), dtype=np.float_)
        if withLabels:
            labelData = np.empty((len(rows),), dtype=np.float_)

        for i, row in enumerate(rows):
            if not row:
                continue
            for j, feature in enumerate(self.features):
                data[i][j] = feature.extractFeature(row)

            if withLabels:
                labelData[i] = self.label.extractFeature(row)

        if withLabels:
            return data, labelData
        else:
            return data

    def setupModel(self, *args, **kwargs):
        raise NotImplementedError

    def fit(self, data, labels):
        raise NotImplementedError

    def runModel(self, data):
        raise NotImplementedError

class Classifier(SupervisedML):
    def testModel(self, data, labels):
        predictions = self.runModel(data)

        return [p == l for p, l in zip(predictions, labels)].count(True)

class DecisionTree(Classifier):
    def __init__(self, features, label, *args, **kwargs):
        super(DecisionTree, self).__init__(features, label, *args, **kwargs)

    def exportGraph(self, dest, labelNames=None):
        dot_data = tree.export_graphviz(self.tree, out_file=None,
                        feature_names=[f.name for f in self.features],
                        class_names=labelNames, rounded=True, filled=True) 
        graph = pydotplus.graph_from_dot_data(dot_data) 
        graph.write_pdf(dest)

    def setupModel(self, *args, **kwargs):
        self.tree = tree.DecisionTreeClassifier(*args, **kwargs)

    def fit(self, data, labels):
        self.tree.fit(data, labels)

    def runModel(self, data):
        return self.tree.predict(data)