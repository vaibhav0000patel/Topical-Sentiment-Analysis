import os
import pickle
from sklearn.svm import NuSVC
from nltk.classify.scikitlearn import SklearnClassifier

def getClassifier(NuSVCPicklePATH,training_set):
    ### NU-SVC CLASSIFIER
    if (os.path.exists(NuSVCPicklePATH)):
        load_NuSVC_classifier = open(NuSVCPicklePATH, "rb")
        NuSVC_classifier = pickle.load(load_NuSVC_classifier)
        load_NuSVC_classifier.close()
        print ("NuSVC_classifier loaded...")
    else:
        NuSVC_classifier = SklearnClassifier(NuSVC())
        NuSVC_classifier.train(training_set)
        save_NuSVC_classifier = open(NuSVCPicklePATH, "wb")
        pickle.dump(NuSVC_classifier, save_NuSVC_classifier)
        save_NuSVC_classifier.close()
        print ("NuSVC_classifier saved...")
        
    return NuSVC_classifier
