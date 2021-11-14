import joblib
import json

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
from sklearn.calibration import CalibratedClassifierCV

# Python script for confusion matrix creation. 
from sklearn.metrics import *
from numpy import mean
from numpy import std
from sklearn import metrics

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

classifier = [
  joblib.load("model_200_no_shuffle.pkl"),
  joblib.load("LinearSVC_no_shuffle.pkl"),
  joblib.load("Random_forest_shuffle.pkl"),
]
thresholds = [
  0.031,
  0.03,
  0.025,
]
hou = 2

testTweets = []
tweet_id_arr = []
dataset_id_arr = []
count_arr = []

cc = 0
for i in range(6):
  df = pd.read_csv(TestDataDir[i], sep='\t')
  cnt = 0
  for row in df.itertuples(name=None):
    cnt += 1
    count_arr.append(cnt)
    tweet_id_arr.append(row[1])
    dataset_id_arr.append(numnum[i])
    testTweets.append(row[2])
    if cc <= 10:
      cc += 1
      print(-22222, row[1], row[2])

X_test = np.array(testTweets)
testLabelPredicted = []
for i in range(len(classifier)):
  testLabelPredicted.append(classifier[i].predict_proba(X_test))

cc = 0
print(classifier[0].classes_)
categories = list(classifier[0].classes_)
for i in range(len(classifier)):
  assert len(classifier[i].classes_) == 24
for i in range(len(classifier[0].classes_)):
  tempx = classifier[0].classes_[i]
  for j in range(len(classifier)):
    assert tempx == classifier[j].classes_[i]

assert len(testLabelPredicted[0]) == len(tweet_id_arr)

import sys
original_stdout = sys.stdout

with open("majority_voting.run", 'w') as output:
  sys.stdout = output
  arrx = []
  for i in range(len(testLabelPredicted[0])):
    temp_dict = {}
    for j in range(len(testLabelPredicted)):
      for k in range(len(testLabelPredicted[j][i])):
        if testLabelPredicted[j][i][k] < thresholds[j]:
          continue
        if diction[categories[k]] in temp_dict:
          temp_dict[diction[categories[k]]] += 1
        else:
          temp_dict[diction[categories[k]]] = 1
    yay = []
    for xx in temp_dict:
      if temp_dict[xx] >= hou:
        yay.append(xx)
    if len(yay) == 0:
      yay = [diction['Irrelevant']]
    arrx.append(yay)

  assert len(arrx) == len(tweet_id_arr)

  for i in range(len(arrx)):
    temp_str = '['
    for j in range(len(arrx[i])):
      if temp_str == '[':
        temp_str += '"' + arrx[i][j] + '"'
      else:
        temp_str += ',"' + arrx[i][j] + '"'
    temp_str += ']'
    print( dataset_id_arr[i], 'Q0', tweet_id_arr[i], count_arr[i], 0.5, temp_str, "myrun", sep = '\t' )

sys.stdout = original_stdout
