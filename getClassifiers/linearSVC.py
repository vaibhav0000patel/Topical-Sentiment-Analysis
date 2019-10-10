import os 
import pickle

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC

def getClassifier(linearSVCPicklePATH,training_set):
    ### LinearSVC CLASSIFIER
    if (os.path.exists(linearSVCPicklePATH)):
        load_LinearSVC_classifier = open(linearSVCPicklePATH, "rb")
        LinearSVC_classifier = pickle.load(load_LinearSVC_classifier)
        load_LinearSVC_classifier.close()
        print ("LinearSVC_classifier loaded...")
    else:
        LinearSVC_classifier = SklearnClassifier(LinearSVC())
        LinearSVC_classifier.train(training_set)
        save_LinearSVC_classifier = open(linearSVCPicklePATH, "wb")
        pickle.dump(LinearSVC_classifier, save_LinearSVC_classifier)
        save_LinearSVC_classifier.close()
        print ("LinearSVC_classifier saved...")
    
    return LinearSVC_classifier