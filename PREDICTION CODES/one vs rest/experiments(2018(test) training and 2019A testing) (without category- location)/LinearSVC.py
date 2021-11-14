import json

diction = {}

diction["GoodsServices"] = "Request-GoodsServices"
diction["SearchAndRescue"] = "Request-SearchAndRescue"
diction["InformationWanted"] = "Request-InformationWanted"
diction["Volunteer"] = "CallToAction-Volunteer"
diction["Donations"] = "CallToAction-Donations"
diction["MovePeople"] = "CallToAction-MovePeople"
diction["FirstPartyObservation"] = "Report-FirstPartyObservation"
diction["ThirdPartyObservation"] = "Report-ThirdPartyObservation"
diction["Weather"] = "Report-Weather"
diction["EmergingThreats"] = "Report-EmergingThreats"
diction["NewSubEvent"] = "Report-NewSubEvent"
diction["MultimediaShare"] = "Report-MultimediaShare"
diction["ServiceAvailable"] = "Report-ServiceAvailable"
diction["Factoid"] = "Report-Factoid"
diction["Official"] = "Report-Official"
diction["News"] = "Report-News"
diction["CleanUp"] = "Report-CleanUp"
diction["Hashtags"] = "Report-Hashtags"
diction["OriginalEvent"] = "Report-OriginalEvent"
diction["ContextualInformation"] = "Other-ContextualInformation"
diction["Advice"] = "Other-Advice"
diction["Sentiment"] = "Other-Sentiment"
diction["Discussion"] = "Other-Discussion"
diction["Irrelevant"] = "Other-Irrelevant"
# diction["Location"] = "Report-Location"


# Import Section
import csv
import codecs
import sys
import io
import numpy as np
import pandas as pd
import scipy as sp

# For Classifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from statistics import mean

# Python script for confusion matrix creation. 
from sklearn.metrics import *
from numpy import mean
from numpy import std
from sklearn import metrics



TrainDataDir="p2. TestSet_2018.csv"
TestDataDir = [
           'p__trecis2019-A-test.earthquakeBohol2013.csv',
           'p__trecis2019-A-test.earthquakeCalifornia2014.csv',
           'p__trecis2019-A-test.fireYMM2016.csv',
           'p__trecis2019-A-test.floodChoco2019.csv',
           'p__trecis2019-A-test.hurricaneFlorence2018.csv',
           'p__trecis2019-A-test.shootingDallas2017.csv'
]

numnum = [
          'TRECIS-CTIT-H-Test-025',
          'TRECIS-CTIT-H-Test-024',
          'TRECIS-CTIT-H-Test-028',
          'TRECIS-CTIT-H-Test-022',
          'TRECIS-CTIT-H-Test-026',
          'TRECIS-CTIT-H-Test-027'
]

def main():
  DataLoad = pd.read_csv(TrainDataDir, sep='\t')
  print(DataLoad.columns)
  print(len(DataLoad.columns))

  ## Define Classifier
  from sklearn.svm import LinearSVC
  from sklearn.svm import SVC

  classifier = Pipeline([
     ('count_vectorizer', CountVectorizer(ngram_range=(1, 3))),
     ('tfidf', TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)),
     ('clf', OneVsRestClassifier(LinearSVC(C=10.0, class_weight=None, dual=True, fit_intercept=True,
      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
      multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
      verbose=0)))]) 
  
  categories = [ "GoodsServices",
          "SearchAndRescue",
          "InformationWanted",
          "Volunteer",
          "Donations",
          "MovePeople",
          "FirstPartyObservation",
          "ThirdPartyObservation",
          "Weather",
          "EmergingThreats",
          "NewSubEvent",
          "MultimediaShare",
          "ServiceAvailable",
          "Factoid",
          "Official",
          "News",
          "CleanUp",
          "Hashtags",
          "OriginalEvent",
          "ContextualInformation",
          "Advice",
          "Sentiment",
          "Discussion",
          "Irrelevant",]
          # "Location"]

  assert len(categories) == 24

  from sklearn.utils import shuffle
  train = shuffle(DataLoad)
  X_train = train['tweet_text']

  testTweets = []
  tweet_id_arr = []
  dataset_id_arr = []
  ans_arr = []
  count_arr = []

  cc = 0
  for i in range(6):
    df = pd.read_csv(TestDataDir[i], sep='\t')
    j = 1
    for row in df.itertuples(name=None):
      tweet_id_arr.append(row[1])
      dataset_id_arr.append(numnum[i])
      testTweets.append(row[2])
      ans_arr.append('[')
      count_arr.append(j)
      j += 1
      if cc <= 10:
        cc += 1
        print(-22222, row[1], row[2])
  
  X_test = np.array(testTweets)

  for category in categories:
    print('... Processing Category: {}'.format(category))
    classifier.fit(X_train, train[category])
    # compute the testing accuracy
    prediction = classifier.predict(X_test)
    assert len(prediction) == len(tweet_id_arr)

    print(-31321, type(prediction))
    for i in range(len(prediction)):
      if prediction[i] == 1:
        if ans_arr[i] == '[':
          ans_arr[i] += '"'
          ans_arr[i] += diction[category]
          ans_arr[i] += '"'
        else:
          ans_arr[i] += ',"'
          ans_arr[i] += diction[category]
          ans_arr[i] += '"'


  import sys
  original_stdout = sys.stdout
  with open("file_linearSVC_12.txt", 'w') as output:
    sys.stdout = output
    for i in range(len(testTweets)):
      ans_arr[i] += ']'
      if ans_arr[i] == '[]':
        ans_arr[i] = '["' + diction["Irrelevant"] + '"]'
      print(dataset_id_arr[i], 'Q0', tweet_id_arr[i], count_arr[i], 0.5, ans_arr[i], 'myrun', sep = '\t' )
  sys.stdout = original_stdout
    
if __name__ == '__main__':
  main()
