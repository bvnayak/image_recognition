import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import utils
import math

# histogram intersection kernel
def histogramIntersection(M, N):
    m = M.shape[0]
    n = N.shape[0]

    result = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            temp = np.sum(np.minimum(M[i], N[j]))
            result[i][j] = temp

    return result


# classify using SVM
def SVM_Classify(trainDataPath, trainLabelPath, testDataPath, testLabelPath, kernelType):
    trainData = np.array(trainDataPath)
    trainLabels = trainLabelPath
    testData = np.array(testDataPath)
    testLabels = testLabelPath

    if kernelType == "HI":

        gramMatrix = histogramIntersection(trainData, trainData)
        clf = SVC(kernel='precomputed')
        clf.fit(gramMatrix, trainLabels)

        predictMatrix = histogramIntersection(testData, trainData)
        SVMResults = clf.predict(predictMatrix)
        correct = sum(1.0 * (SVMResults == testLabels))
        accuracy = correct / len(testLabels)

        print "SVM (Histogram Intersection): " +str(accuracy * 100)+ "% (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"

    else:
        clf = SVC(kernel=kernelType)
        clf.fit(trainData, trainLabels)
        SVMResults = clf.predict(testData)

        correct = sum(1.0 * (SVMResults == testLabels))
        accuracy = correct / len(testLabels)
        print "SVM (" +kernelType+"): " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"

    createConfusionMatrix(SVMResults, testLabels, kernelType)


def createConfusionMatrix(results, labels, kernel_type):
    cm = confusion_matrix(labels, results)

    np.set_printoptions(precision=2)
    print('Confusion matrix')
    print cm

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion_matrix")
    plt.colorbar()
    unique_labels = np.unique(labels)
    tick_marks = np.arange(len(unique_labels))
    plt.xticks(tick_marks, unique_labels, rotation=45)
    plt.yticks(tick_marks, unique_labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("Confusion_matrix.png", format="png", dpi=600)
