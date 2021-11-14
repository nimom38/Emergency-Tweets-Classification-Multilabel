
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



TrainDataDir="2. TestSet_2018.csv"
TestDataDir = [
           '__trecis2019-A-test.earthquakeBohol2013.csv',
           '__trecis2019-A-test.earthquakeCalifornia2014.csv',
           '__trecis2019-A-test.fireYMM2016.csv',
           '__trecis2019-A-test.floodChoco2019.csv',
           '__trecis2019-A-test.hurricaneFlorence2018.csv',
           '__trecis2019-A-test.shootingDallas2017.csv'
]

numnum = [
          'TRECIS-CTIT-H-Test-025',
          'TRECIS-CTIT-H-Test-024',
          'TRECIS-CTIT-H-Test-028',
          'TRECIS-CTIT-H-Test-022',
          'TRECIS-CTIT-H-Test-026',
          'TRECIS-CTIT-H-Test-027'
]

frames = []

for i in range(6):
  files = pd.read_csv(TestDataDir[i])
  frames.append(files)
  #print(files.columns)

result = pd.concat(frames, ignore_index=True)
result.set_index('identifier', inplace=True, drop=True)

cc = 0
for row in result.itertuples(name=None):
  cc += 1
  #print(row)

w, h = 1, cc
Matrix = [[0 for x in range(w)] for y in range(h)]
lol = []

def main():
  DataLoad = pd.read_csv(TrainDataDir)
  
  #df_label = DataLoad.drop(['AnimeName', 'SearchContents'], axis=1)
  #counts = []
  #categories = list(df_label.columns.values)
  #for i in categories:
    #counts.append((i, df_label[i].sum()))
  #df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])
  #print(df_stats)
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

  X_train = train.tweet_text
  #print(X_train.shape)
  #print(X_test.shape)

  ## Define Classifier
  from sklearn.ensemble import RandomForestClassifier

  classifier = Pipeline([
     ('count_vectorizer', CountVectorizer(ngram_range=(1, 3))),
     ('clf', OneVsRestClassifier(RandomForestClassifier(max_depth=2, random_state=0)))])


  df_predicted=pd.DataFrame()
  df_groundTruth=pd.DataFrame()
  ok = False
  for category in categories:
    print('... Processing Category: {}'.format(category))
    # train the model using X_dtm & y
    classifier.fit(X_train, train[category])
    cnt = 0
    for i in range(6):
      huhu = pd.read_csv(TestDataDir[i])
      test = huhu
      X_test = test.tweet_text
      #compute the testing accuracy
      prediction = classifier.predict(X_test)
      for j in range(len(prediction)):
        if prediction[j] == 1:
          Matrix[cnt].append('"' + diction[category] + '"')
        cnt += 1
        if ok == False:
          lol.append( (numnum[i], j+1) )
    ok = True
      #print(prediction)
      #print(test[category])
      #df_predicted[category]=prediction
      #df_groundTruth[category]=test[category]
      #print('F1 Micro: {}'.format(f1_score(test[category], prediction, average='micro')))

  cnt = 0
  arr = []
  for row in result.itertuples(name=None):
    temp = []
    temp.append(lol[cnt][0])
    temp.append("Q0")
    temp.append(row[0])
    temp.append(lol[cnt][1])
    temp.append(1)
    Matrix[cnt].pop(0)
    temp.append(Matrix[cnt])
    temp.append("myrun")
    cnt += 1
    arr.append(temp)
  
  with open("file_Random_Forest_Classifier.txt", 'w') as output:
    for row in arr:
        output.write(str(row) + '\n')

  print(result.shape)

  #y_true = np.array(df_groundTruth)
  #y_pred = np.array(df_predicted)

  #print("\n")
  #print("F1_Micro:", f1_score(y_true, y_pred, average='micro'))
  #print("F1_Macro:", f1_score(y_true, y_pred, average='macro'))
  #print("Multi-label Accuracy (or Jaccard Index):", jaccard_score(y_true,y_pred, average='samples'))
  #print("Hamming_loss:", hamming_loss(y_true, y_pred))    
  
  
if __name__ == '__main__':
  main()
