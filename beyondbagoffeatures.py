from recognition import utils
from recognition import classification
from recognition.visualVocabulary import Vocabulary

folderPath = "images/"


def buildHistogram(path, trainData, voc, training, level, sift):
    if training is False:
        trainData = utils.readImages(folderPath + path, sift)[0]

    # Transform each feature into histogram
    featureHistogram = []
    labels = []

    index = 0
    for oneImage in trainData:
        featureHistogram.append(voc.buildHistogramForEachImageAtDifferentLevels(oneImage, level))
        labels.append(oneImage.label)

        index += 1

    return [featureHistogram, labels]


def main():
    level = 2
    sift = False

    training_path = "c1_test"
    testing_path = "c1_train"

    # training_path = "training"
    # testing_path = "testing"

    # training_path = "caltech_train"
    # testing_path = "caltech_test"


    # read train data
    [train_data, train_features] = utils.readImages(folderPath + training_path, sift)
    # create vocabulary
    training_voc = Vocabulary(train_features, 200)

    [test_hist, test_label] = buildHistogram(testing_path, None, training_voc, False, level, sift)
    [train_hist, train_label] = buildHistogram(training_path, train_data, training_voc, True, level, sift)

    classification.SVM_Classify(train_hist, train_label, test_hist, test_label, "HI")


if __name__ == "__main__":

    main()




