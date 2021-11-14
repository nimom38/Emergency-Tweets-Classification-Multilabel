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
  
  X_train = np.array( DataLoad['tweet_text'] )
  Y_train = np.array( DataLoad['XXXX'] )

  from sklearn.utils import shuffle

  from sklearn.neural_network import MLPClassifier
  classifier = Pipeline([
     ('count_vectorizer', CountVectorizer(ngram_range=(1, 3))),
     ('clf', MLPClassifier(hidden_layer_sizes=(50, 50, 50, 50), random_state=1, max_iter=1000, warm_start=True))])

  classifier.fit(X_train, Y_train)

  joblib.dump(classifier, "model_50_50_50_50_no_shuffle.pkl") 

  testTweets = []
  tweet_id_arr = []
  dataset_id_arr = []

  cc = 0
  for i in range(6):
    df = pd.read_csv(TestDataDir[i], sep='\t')
    for row in df.itertuples(name=None):
      tweet_id_arr.append(row[1])
      dataset_id_arr.append(numnum[i])
      testTweets.append(row[2])
      if cc <= 10:
        cc += 1
        print(-22222, row[1], row[2])
  
  X_test = np.array(testTweets)
  testLabelPredicted = classifier.predict_proba(X_test)

  cc = 0
  print(classifier.classes_)
  for eachProb in testLabelPredicted:
    if cc <= 10:
      cc += 1
      print (eachProb)
  
if __name__ == '__main__':
  main()