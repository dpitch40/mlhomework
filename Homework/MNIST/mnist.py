import csv
import operator
import numpy as np
import theano
import theano.tensor as T
import lasagne
import itertools

NUM_PIXELS = 784
NUM_TRAINING_STEPS = 10
NUM_INPUT_ROWS = 5000

def loadData(fName, training=True, limit=None):
    if training:
        labels = list()
    data = list()

    with open(fName, 'r') as fObj:
        reader = csv.reader(fObj)
        headers = map(operator.methodcaller("lower"), reader.next())
        if limit:
            itr = itertools.islice(reader, limit)
        else:
            itr = reader
        for row in itr:
            if not row:
                continue
            if training:
                labels.append(row[0])
                row = row[1:]
            data.append(row)

    print "Loaded %s" % fName
    dataArr = np.array(data, dtype=np.double) / 256.0
    if training:
        labelsArr = np.array(labels, dtype=np.uint8)
        return labelsArr, dataArr
    else:
        return dataArr

def buildNetwork(inputVar, numInstances=None):
    lIn = lasagne.layers.InputLayer(shape=(numInstances, NUM_PIXELS),
                                    input_var=inputVar)

    lDrop1 = lasagne.layers.DropoutLayer(lIn, p=0.2)

    lHid1 = lasagne.layers.DenseLayer(lDrop1, num_units=800,
                        nonlinearity=lasagne.nonlinearities.rectify,
                        W=lasagne.init.GlorotUniform())
    
    lDrop2 = lasagne.layers.DropoutLayer(lHid1, p=0.5)

    lHid2 = lasagne.layers.DenseLayer(lDrop2, num_units=800,
                        nonlinearity=lasagne.nonlinearities.rectify,
                        # W=lasagne.init.GlorotUniform()
                        )
    
    lDrop3 = lasagne.layers.DropoutLayer(lHid2, p=0.5)

    lOut = lasagne.layers.DenseLayer(lDrop3, num_units=10,
                        nonlinearity=lasagne.nonlinearities.softmax)

    return lOut

def main():
    labels, trainingData = loadData("train.csv", limit=NUM_INPUT_ROWS)
    numInstances = labels.shape[0]
    assert numInstances == trainingData.shape[0]
    assert trainingData.shape[1] == NUM_PIXELS

    inputVar = T.dmatrix(name="training")
    outputVar = T.ivector(name="targets")
    network = buildNetwork(inputVar, numInstances)

    prediction = lasagne.layers.get_output(network)
    val_fn = theano.function([inputVar], prediction)
    loss = lasagne.objectives.categorical_crossentropy(prediction, outputVar)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    train_fn = theano.function([inputVar, outputVar], loss, updates=updates)

    for step in xrange(NUM_TRAINING_STEPS):
        trainErr = train_fn(trainingData, labels)
        print "Completed step %d: error = %s" % (step, trainErr)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            outputVar)
    test_loss = test_loss.mean()

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), outputVar),
                  dtype=theano.config.floatX)

    acc_fn = theano.function([inputVar, outputVar], [test_acc, test_loss])

    print acc_fn(trainingData, labels)


if __name__ == "__main__":
    main()