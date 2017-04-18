import operator

class Feature(object):
    def __init__(self, name, retriever, converter=float, defaultVal=None):
        self.name = name
        self.retriever = retriever
        self.converter = converter
        self.defaultVal = defaultVal

    def extractFeature(self, inputRow):
        inputVal = self.retriever(inputRow)
        if self.converter:
            try:
                return self.converter(inputVal)
            except:
                return self.defaultVal
        else:
            return inputVal

class LookupFeature(Feature):
    def __init__(self, name, converter=float, lookupKey=None, **kwargs):
        if lookupKey:
            key = lookupKey
        else:
            key = name
        retriever = operator.itemgetter(key)
        super(LookupFeature, self).__init__(name, retriever, converter, **kwargs)