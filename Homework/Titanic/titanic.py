from sklearn import tree
import numpy as np
import csv
import pydotplus
import operator

import sys
import os
sys.path.append(os.getcwd())

from Feature import LookupFeature
from MLFramework import DecisionTree

genderMap = {"male": 1, "female": 2}

features = (LookupFeature("Class", lookupKey="Pclass"),
            LookupFeature("Sex", converter=lambda x: genderMap[x]),
            LookupFeature("Age", defaultVal=29.7),
            LookupFeature("SibSp"),
            LookupFeature("Parch"),
            LookupFeature("Fare"))

label = LookupFeature("Survived")

def main():
    with open("Titanic/train.csv") as fObj:
        tree = DecisionTree(features, label, max_leaf_nodes=20)
        trainingData, labels = tree.loadTrainingData(csv.DictReader(fObj))
        tree.fit(trainingData, labels)
        # tree.exportGraph("Titanic.pdf", ["Dead", "Survived"])

        matched = tree.testModel(trainingData, labels)
        print "Matched %d / %d" % (matched, labels.shape[0])

if __name__ == "__main__":
    main()