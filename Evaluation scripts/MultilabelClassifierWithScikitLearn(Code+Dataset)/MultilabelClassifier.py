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


TrainDataDir="/content/drive/My Drive/Colab Notebooks/AnimeGenreAnalysis/MultiLabelAnimeDatasetProcessed.txt"

def main():
  DataLoad= pd.read_csv(TrainDataDir,sep='\t',skipinitialspace=False, quoting=csv.QUOTE_NONE)
  
  df_label = DataLoad.drop(['AnimeName', 'SearchContents'], axis=1)
  counts = []
  categories = list(df_label.columns.values)
  for i in categories:
    counts.append((i, df_label[i].sum()))
  df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])
  #print(df_stats)
  categories = ['comedy', 'shounen', 'drama', 'action', 'adventure', 'fantasy', 'scifi', 'romance', 'supernatural', 'mystery', 'sliceoflife', 'school', 'seinen', 'historical', 'sports']
  
  train, test = train_test_split(DataLoad, random_state=42, test_size=0.2, shuffle=True)

  X_train = train.SearchContents
  X_test = test.SearchContents
  #print(X_train.shape)
  #print(X_test.shape)


  ## Define Classifier
  classifier = Pipeline([
     ('count_vectorizer', CountVectorizer(ngram_range=(1,3))),
     ('tfidf', TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)),
     ('clf', OneVsRestClassifier(MultinomialNB()))])  


  df_predicted=pd.DataFrame()
  df_groundTruth=pd.DataFrame()
  for category in categories:
    print('... Processing Category: {}'.format(category))
    # train the model using X_dtm & y
    classifier.fit(X_train, train[category])
    
	# compute the testing accuracy
    prediction = classifier.predict(X_test)
    #print(prediction)
    df_predicted[category]=prediction
    df_groundTruth[category]=test[category]
    print('F1 Micro: {}'.format(f1_score(test[category], prediction, average='micro')))

  
  #print(df_predicted)
  #print(df_groundTruth)

  y_true = np.array(df_groundTruth)
  y_pred = np.array(df_predicted)

  print("\n")
  print("F1_Micro:", f1_score(y_true, y_pred, average='micro'))
  print("F1_Macro:", f1_score(y_true, y_pred, average='macro'))
  print("Multi-label Accuracy (or Jaccard Index):", jaccard_score(y_true,y_pred, average='samples'))
  print("Hamming_loss:", hamming_loss(y_true, y_pred))    
  
  
if __name__ == '__main__':
  main()