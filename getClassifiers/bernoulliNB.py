import os
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB

def getClassifier(bernoulliNBPicklePATH,training_set):
    ### BERNOULLI NAIVE BAYES CLASSIFIER
    if (os.path.exists(bernoulliNBPicklePATH)):
        load_BernoulliNB_classifier5k = open(bernoulliNBPicklePATH, "rb")
        BernoulliNB_classifier = pickle.load(load_BernoulliNB_classifier5k)
        load_BernoulliNB_classifier5k.close()
        print ("BernoulliNB_classifier loaded...")
    else:
        BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
        BernoulliNB_classifier.train(training_set)
        save_BernoulliNB_classifier5k = open(bernoulliNBPicklePATH, "wb")
        pickle.dump(BernoulliNB_classifier, save_BernoulliNB_classifier5k)
        save_BernoulliNB_classifier5k.close()
        print ("BernoulliNB_classifier saved...")
    return BernoulliNB_classifier