import os
import pickle

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression

def getClassifier(logisticRegressionPicklePATH,training_set):
    ### LOGISTIC REGRESSION CLASSIFIER
    if (os.path.exists(logisticRegressionPicklePATH)):
        load_LogisticRegression_classifier = open(logisticRegressionPicklePATH, "rb")
        LogisticRegression_classifier = pickle.load(load_LogisticRegression_classifier)
        load_LogisticRegression_classifier.close()
        print ("LogisticRegression_classifier loaded...")
    else:
        LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
        LogisticRegression_classifier.train(training_set)
        save_LogisticRegression_classifier = open(logisticRegressionPicklePATH, "wb")
        pickle.dump(LogisticRegression_classifier, save_LogisticRegression_classifier)
        save_LogisticRegression_classifier.close()
        print ("LogisticRegression_classifier saved...")

    return LogisticRegression_classifier