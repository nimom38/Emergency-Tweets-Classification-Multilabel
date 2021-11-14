# -*- coding: utf-8 -*-
"""training(2018-test) and testing(2019A) (Linear SVC (SVM)).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DknaobiyGL9UqTSsnZTZ4hOL84wCSg2w
"""

import json

# #### NLTK install
# !pip install --user -U nltk

# ### Krovetzstemmer install
# !pip install krovetzstemmer

# ### spacy install
# !pip install -U spacy
# !python -m spacy download en_core_web_sm

# ### URL Normalization install
# !pip install urlextract
# !pip install idna
# !pip install uritools
# !pip install appdirs
# !pip install dnspython

# ### install language detection
# !pip install langdetect

# ### install ekphrasis
# !pip install ekphrasis

from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from krovetzstemmer import Stemmer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from gensim.parsing.preprocessing import STOPWORDS
import spacy
from urlextract import URLExtract
from langdetect import detect
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

######### TOKENIZATION ##########

## NLTK
tknzr = TweetTokenizer()
def NLTK_TOKENIZE_TWEET(text):
  # from nltk.tokenize import TweetTokenizer
  temp = tknzr.tokenize(text)
  s = " "   
  return (s.join(temp))

######### STEMMING ##########

## PORTER STEMMER
# from nltk.stem.porter import *
stemmer_p = PorterStemmer()
def PORTER_STEMMER(text):
  li = list(text.split(" "))
  singles = [stemmer_p.stem(xx) for xx in li]
  return (' '.join(singles))

## SNOWBALL STEMMER
stemmer_s = SnowballStemmer(language='english', ignore_stopwords=True) #ignore_stopwords=False to disable ignoring stopwords
def SNOWBALL_STEMMER(text):
  li = list(text.split(" "))
  # from nltk.stem.snowball import SnowballStemmer
  singles = [stemmer_s.stem(xx) for xx in li]
  return (' '.join(singles))

## KROVETZ STEMMER
stemmer_k = Stemmer()
def KROVETZ_STEMMER(text):
  li = list(text.split(" "))
  # from krovetzstemmer import Stemmer
  singles = [stemmer_k.stem(xx) for xx in li]
  return (' '.join(singles))

######### STOPWORDS REMOVAL ##########

## NLTK
all_stopwords_n = stopwords.words('english')
def NLTK_STOPWORD(text):
  # from nltk.corpus import stopwords
  # import nltk
  # nltk.download('stopwords')

  ###### add new stopwords
  # sw_list = ['likes','play']
  # all_stopwords_n.extend(sw_list)
  ######

  ###### remove stopwords
  # all_stopwords_n.remove('not')
  ######

  text_tokens = list(text.split(" "))
  tokens_without_sw = [word for word in text_tokens if not word in all_stopwords_n]
  return (' '.join(tokens_without_sw))

## Genism
all_stopwords_gensim = STOPWORDS
def GENISM_STOPWORD(text):
  # from gensim.parsing.preprocessing import STOPWORDS
  
  ##### add new stopwords
  # all_stopwords_gensim = STOPWORDS.union(set(['likes', 'play']))
  #####

  ##### remove stopwords
  # sw_list = {"not"}
  # all_stopwords_gensim = STOPWORDS.difference(sw_list)
  #####

  text_tokens = list(text.split(" "))
  tokens_without_sw = [word for word in text_tokens if not word in all_stopwords_gensim]
  return (' '.join(tokens_without_sw))

## SpacY
sp = spacy.load('en_core_web_sm')
all_stopwords_s = sp.Defaults.stop_words
def SPACY_STOPWORD(text):
  # import spacy
  
  ##### add stopwords
  # all_stopwords_s |= {"likes","tennis",}
  #####

  ##### remove stopwords
  # all_stopwords_s.remove('not')
  #####

  text_tokens = list(text.split(" "))
  tokens_without_sw = [word for word in text_tokens if not word in all_stopwords_s]
  return (' '.join(tokens_without_sw))

##### SPECIAL CHARACTERS REMOVAL #####

## Also removes # and @ (mind you)
def special_characters_removal(text):
  hmm = ""
  for ch in text:
    if ch.isalnum():
      hmm += ch
    elif ch == ' ':
      hmm += ch
  return hmm

# print( special_characters_removal('dnjaskd% asjdhibas&j asjdbijb**###@??>< dbiasbdasib.!') )

##### URL NORMALIZATION #####
extractor = URLExtract()
def url_normalization(text):
  text_tokens = list(text.split(" "))
  # from urlextract import URLExtract
  for i in range(len(text_tokens)):
    example_text = text_tokens[i]
    if extractor.has_urls(example_text):
      text_tokens[i] = "url"
  return (' '.join(text_tokens))

# print( url_normalization('hihihihi facebook.com yahoo.com http://bit.ly/PdHur https://tinyurl.com/uxtct20') )

##### Language Detection #####

def lang_detection(text):
  # from langdetect import detect
  return detect(text)

# this function is expected to return "en" for english
# print(lang_detection('this is english. i speak english. i speak bangla'))

##### ekphrasis pipeline #####

def ekphrasis_pipeline(text):
  # from ekphrasis.classes.preprocessor import TextPreProcessor
  # from ekphrasis.classes.tokenizer import SocialTokenizer
  # from ekphrasis.dicts.emoticons import emoticons

  text_processor = TextPreProcessor(
    # terms that will be normalized
    # normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        # 'time', 'url', 'date', 'number'],
    # terms that will be annotated
    # annotate={"hashtag", "allcaps", "elongated", "repeated",
        # 'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    # tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    # dicts=[emoticons]
  )
  return (" ".join(text_processor.pre_process_doc(text)))

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
ans = []
sentences = []

text_processor = TextPreProcessor(
    # terms that will be normalized
    # normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
    #     'time', 'url', 'date', 'number'],
    # # terms that will be annotated
    # annotate={"hashtag", "allcaps", "elongated", "repeated",
    #     'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    # dicts=[emoticons]
  )
def mass_ekphrasis():
  # print("len: ", len(sentences))
  ans.clear()
  # from ekphrasis.classes.preprocessor import TextPreProcessor
  # from ekphrasis.classes.tokenizer import SocialTokenizer
  # from ekphrasis.dicts.emoticons import emoticons

  for s in sentences:
    ans.append(" ".join(text_processor.pre_process_doc(s)))
    # ans.append(s)
    # print(" ".join(text_processor.pre_process_doc(s)), -1121321, s)
  
  #return ans

def preprocess_pipeline( text ):
  # text = ekphrasis_pipeline(text)
  text = NLTK_TOKENIZE_TWEET(text)
  text = PORTER_STEMMER(text)
  text = NLTK_STOPWORD(text)
  text = special_characters_removal(text)
  text = url_normalization(text)
  return text

def preprocess_preprocess(TT, hmm):
  arr = [ TT[0], TT[1] ]
  # print(hmm)
  temp = preprocess_pipeline( hmm )
  arr.append( temp )
  for i in range(3, len(TT)):
    arr.append( TT[i] )
  # print(TT[2], temp)
  return tuple(arr)

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
  # train = shuffle(DataLoad)

  ##########################################################################
  # print(DataLoad.columns)
  arrr = []
  sentences.clear()
  lol_cnt = 0
  damnit = []
  for row in DataLoad.itertuples(name=None):
    if lang_detection(row[2]) == 'en':
      sentences.append(row[2])
      damnit.append(1)
    else:
      damnit.append(0)
  # print(-11111111, len(sentences))
  mass_ekphrasis()
  # print(-22222222, len(sentences), type(sentences))
  iii = -1
  for row in DataLoad.itertuples(name=None):
    # print(row)
    iii += 1
    if damnit[iii] == 1:
      # print(-1321312, ans[lol_cnt], -123123123, row[2])
      arrr.append( preprocess_preprocess(row, ans[lol_cnt]) )
      lol_cnt += 1
  assert lol_cnt == len(ans)
  assert lol_cnt == len(sentences)
  yoyoyo = ['0']
  for hmm in DataLoad.columns:
    yoyoyo.append(hmm)
  train = pd.DataFrame(arrr, columns = yoyoyo)
  # train.set_index('identifier', drop = True, inplace = True)
  train = train.drop(['0'], axis = 1)
  # print(train.columns)
  # print(train.index)
  # print(DataLoad.index)
  # print(train.shape)
  # print(DataLoad.shape)
  # print("---------------------------------------------------------")
  #########################################################################

  X_train = train.tweet_text

  # print(type(X_train))

  #print(X_train.shape)
  #print(X_test.shape)

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


  df_predicted=pd.DataFrame()
  df_groundTruth=pd.DataFrame()
  ok = False
  for category in categories:
    print('... Processing Category: {}'.format(category))
    # train the model using X_dtm & y
    # print(X_train)
    # print(train[category])
    classifier.fit(X_train, train[category])
    cnt = 0
    for i in range(6):
      huhu = pd.read_csv(TestDataDir[i])

      #########################################################################
      arrr = []
      sentences.clear()
      lol_cnt = 0
      for row in huhu.itertuples(name=None):
        sentences.append(row[2])
      mass_ekphrasis()
      for row in huhu.itertuples(name=None):
        # print(row)
        arrr.append( preprocess_preprocess(row, ans[lol_cnt]) )
        lol_cnt += 1
      assert lol_cnt == len(ans)
      assert lol_cnt == len(sentences)
      yoyoyo = ['0']
      for hmm in huhu.columns:
        yoyoyo.append(hmm)
      test = pd.DataFrame(arrr, columns = yoyoyo)
      # train.set_index('identifier', drop = True, inplace = True)
      test = test.drop(['0'], axis = 1)

      # print(huhu.columns)
      # print(test.columns)
      # print(huhu.shape)
      # print(test.shape)
      # print(huhu.index)
      # print(test.index)

      ###############################################################################
      X_test = test.tweet_text
      # print(X_test)
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
  
  with open("file_linearSVC_1.txt", 'w') as output:
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