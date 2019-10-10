import os
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import SGDClassifier

def getClassifier(SGDClassifierPicklePATH,training_set):
    ### SGD CLASSIFIER
    if (os.path.exists(SGDClassifierPicklePATH)):
        load_SGDClassifier_Classifier = open(SGDClassifierPicklePATH, "rb")
        SGDClassifier_classifier = pickle.load(load_SGDClassifier_Classifier)
        load_SGDClassifier_Classifier.close()
        print ("SGDClassifier_classifier loaded...")
    else:
        SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
        SGDClassifier_classifier.train(training_set)
        save_SGDClassifier_classifier = open(SGDClassifierPicklePATH, "wb")
        pickle.dump(SGDClassifier_classifier, save_SGDClassifier_classifier)
        save_SGDClassifier_classifier.close()
        print ("SGDClassifier_classifier saved...")

    return SGDClassifier_classifier