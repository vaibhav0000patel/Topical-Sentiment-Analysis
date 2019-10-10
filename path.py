dir_name = "pickled_classifiers/"

naiveBayesPickle        = 'originalnaivebayes.pickle'
multinomialNBPickle     = 'MNB_classifier.pickle'
bernoulliNBPickle       = 'BernoulliNB_classifier.pickle'
logisticRegressionPickle= 'LogisticRegression_classifier.pickle'
linearSVCPickle         = 'LinearSVC_classifier.pickle'
SGDClassifierPickle     = 'SGDC_classifier.pickle'
NuSVCPickle             = 'NuSVC_classifier.pickle'

naiveBayesPicklePATH            = dir_name + naiveBayesPickle
multinomialNBPicklePATH         = dir_name + multinomialNBPickle
bernoulliNBPicklePATH           = dir_name + bernoulliNBPickle
logisticRegressionPicklePATH    = dir_name + logisticRegressionPickle
linearSVCPicklePATH             = dir_name + linearSVCPickle
SGDClassifierPicklePATH         = dir_name + SGDClassifierPickle
NuSVCPicklePATH                 = dir_name + NuSVCPickle

pickleDataDir_name  = "pickled_data/"

datasetPath         = "dataset/labeled_tweets.txt"
documentsPath       = pickleDataDir_name+"documents.pickle"
datasetTweetsPath   = pickleDataDir_name+"datasetTweets.pickle"
wordFeaturesPath    = pickleDataDir_name+"wordFeatures.pickle"
featuresetsPath     = pickleDataDir_name+"featuresets.pickle"
