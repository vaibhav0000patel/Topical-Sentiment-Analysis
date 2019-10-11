from __future__ import division
import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
from nltk.classify import ClassifierI
from nltk.corpus import stopwords

from statistics import mode

from getClassifiers import bernoulliNB,linearSVC,logisticRegression,multinomialNB,naiveBayes,nuSVC,SGDClassifier

import pickle
import path
import os
import re

tknzr = TweetTokenizer()
stopWords = set(stopwords.words("english"))

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes=[]
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = float(choice_votes / len(votes))
        return conf

def processTweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    tweet = re.sub(r'@[^\s]+','AT_USER',tweet)
    tweet = re.sub(r'[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.strip('\'"')
    return tweet

def find_features(document,wordFeatures):
    words = set(document)
    features = {}
    for w in wordFeatures:
        features[w] = (w in words)
    return features

def prepareDocuments(datasetPath,documentsPath,datasetTweetsPath):

    if (os.path.exists(documentsPath) and os.path.exists(datasetTweetsPath)):
        
        load_documents = open(documentsPath, "rb")
        documents = pickle.load(load_documents)
        load_documents.close()
        print("documents loaded...")

        load_datasetTweets = open(datasetTweetsPath, "rb")
        datasetTweets = pickle.load(load_datasetTweets)
        load_datasetTweets.close()
        print("datasetTweets loaded...")

    else:

        dataset = open(datasetPath,"r",encoding='utf8').read()
        documents = []
        datasetTweets = []

        for r in dataset.split('\n'):
            processedTweet = []
            for doc_word in tknzr.tokenize(processTweet(r[2:])):
                processedTweet.append(doc_word.lower())
            if(len(r)>0 and r[0]=='1'):
                documents.append((processedTweet, "related"))
            elif(len(r)>0 and r[0]=='0'):
                documents.append((processedTweet, "unrelated"))
            datasetTweets.append(processedTweet)

        save_documents = open(documentsPath, "wb") 
        pickle.dump(documents, save_documents)
        save_documents.close()
        print("documents saved...")

        save_datasetTweets = open(datasetTweetsPath, "wb") 
        pickle.dump(datasetTweets, save_datasetTweets)
        save_datasetTweets.close()
        print("datasetTweets saved...")

    return documents,datasetTweets

def prepareWordFeatures(datasetTweets,wordFeaturesPath):

    if (os.path.exists(wordFeaturesPath)):

        load_wordFeatures = open(wordFeaturesPath, "rb")
        wordFeatures = pickle.load(load_wordFeatures)
        load_wordFeatures.close()
        print("wordFeatures loaded...")

    else:

        all_words = []
        for dataset_tweet in datasetTweets:
            for word in dataset_tweet:
                if word not in stopWords:
                    all_words.append(word.lower())

        all_words = nltk.FreqDist(all_words)
        wordFeatures = list(all_words.keys())

        save_wordFeatures = open(wordFeaturesPath, "wb")
        pickle.dump(wordFeatures, save_wordFeatures)
        save_wordFeatures.close()
        print("wordFeatures saved...")

    return wordFeatures

def prepareFeaturesets(documents,wordFeatures,featuresetsPath):
    
    if (os.path.exists(featuresetsPath)):
        
        load_featuresets = open(featuresetsPath, "rb") 
        featuresets = pickle.load(load_featuresets)
        load_featuresets.close()
        print("featuresets loaded...")

    else:
        
        featuresets = [(find_features(rev,wordFeatures),category) for (rev,category) in documents]

        save_featuresets = open(featuresetsPath, "wb")
        pickle.dump(featuresets, save_featuresets)
        save_featuresets.close()
        print("featuresets saved...")
    
    return featuresets

def prepareClassifier():

    documents,datasetTweets = prepareDocuments(path.datasetPath,path.documentsPath,path.datasetTweetsPath)
    wordFeatures = prepareWordFeatures(datasetTweets,path.wordFeaturesPath)
    training_set = prepareFeaturesets(documents,wordFeatures,path.featuresetsPath)

    classifier                      = naiveBayes.getClassifier(path.naiveBayesPicklePATH,training_set)
    MultinomialNB_classifier        = multinomialNB.getClassifier(path.multinomialNBPicklePATH,training_set)
    BernoulliNB_classifier          = bernoulliNB.getClassifier(path.bernoulliNBPicklePATH,training_set)
    LogisticRegression_classifier   = logisticRegression.getClassifier(path.logisticRegressionPicklePATH,training_set)
    LinearSVC_classifier            = linearSVC.getClassifier(path.linearSVCPicklePATH,training_set)
    SGDClassifier_classifier        = SGDClassifier.getClassifier(path.SGDClassifierPicklePATH,training_set)
    NuSVC_classifier                = nuSVC.getClassifier(path.NuSVCPicklePATH,training_set)

    #testing_set = []
    #print "Original NaiveBayes Accuracy : ",(nltk.classify.accuracy(classifier,testing_set)*100)
    #print classifier.show_most_informative_features(15)
    #print "MultinomialNB_classifier Accuracy : ",(nltk.classify.accuracy(MultinomialNB_classifier,testing_set)*100)
    #print "BernoulliNB_classifier Accuracy : ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set) * 100)
    #print "LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier,testing_set)) * 100
    #print "LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100
    #print "SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier,testing_set)) * 100
    #print "NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier,testing_set)) * 100

    voted_classifier = VoteClassifier(  classifier,
                                        MultinomialNB_classifier,
                                        BernoulliNB_classifier,
                                        LogisticRegression_classifier,
                                        SGDClassifier_classifier,
                                        LinearSVC_classifier,
                                        NuSVC_classifier
                                    )
    return voted_classifier

def relatibility(text):

    voted_classifier = prepareClassifier()
    wordFeatures = prepareWordFeatures(None,path.wordFeaturesPath)
    feats = find_features([word.lower() for word in tknzr.tokenize(processTweet(text)) if word not in stopWords],wordFeatures)
    print ("")
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)
