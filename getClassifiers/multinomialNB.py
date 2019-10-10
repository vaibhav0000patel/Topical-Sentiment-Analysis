import pickle
import os
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB

def getClassifier(multinomialNBPicklePATH,training_set):

    ### MULTI-NOMIAL NAIVE BAYES CLASSIFIER
    if (os.path.exists(multinomialNBPicklePATH)):
        load_MultinomialNB_classifier = open(multinomialNBPicklePATH, "rb")
        MultinomialNB_classifier = pickle.load(load_MultinomialNB_classifier)
        load_MultinomialNB_classifier.close()
        print ("MultinomialNB_classifier loaded...")
    else:
        MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
        MultinomialNB_classifier.train(training_set)
        save_MultinomialNB_classifier = open(multinomialNBPicklePATH, "wb")
        pickle.dump(MultinomialNB_classifier, save_MultinomialNB_classifier)
        save_MultinomialNB_classifier.close()
        print ("MultinomialNB_classifier saved...")
        
    return MultinomialNB_classifier