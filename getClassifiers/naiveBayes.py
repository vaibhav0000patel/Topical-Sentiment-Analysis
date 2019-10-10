import os
import pickle
import nltk

def getClassifier(naiveBayesPicklePATH,training_set):
    ### NAIVE BAYES CLASSIFIER
    if(os.path.exists(naiveBayesPicklePATH)):
        load_classifier = open(naiveBayesPicklePATH, "rb")
        classifier = pickle.load(load_classifier)
        load_classifier.close()
        print ("NAIVE-BAYES CLASSIFIER loaded...")
    else:
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        save_classifier = open(naiveBayesPicklePATH, "wb")
        pickle.dump(classifier, save_classifier)
        save_classifier.close()
        print ("NAIVE-BAYES CLASSIFIER saved...")
        
    return classifier