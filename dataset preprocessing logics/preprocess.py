import json
import pandas as pd

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
from ekphrasis.classes.segmenter import Segmenter
seg = Segmenter(corpus="twitter")

######### HASHTAG SEGMENTATION ########

def hashtagSegmentation(text):
  givenText=text
  getHashTagFromText = [t for t in givenText.split() if t.startswith('#')]
  segment_the_Hash =''
  getHashtagSegmentedText = givenText
  if getHashTagFromText:
      for eachHashTag in getHashTagFromText:
           eachHashTag=eachHashTag[1:]
           segment_the_Hash = seg.segment(eachHashTag)
           getHashtagSegmentedText = getHashtagSegmentedText+" "+ segment_the_Hash

  return getHashtagSegmentedText

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

print(KROVETZ_STEMMER('utilities'))

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

def preprocess_pipeline( text ):
  text = hashtagSegmentation(text)
  text = url_normalization(text)
  text = NLTK_TOKENIZE_TWEET(text)
  text = KROVETZ_STEMMER(text)
  # text = PORTER_STEMMER(text)
  text = NLTK_STOPWORD(text)
  text = special_characters_removal(text)
  return text

TrainDataDir = [
           "2. TestSet_2018.csv"
]
TestDataDir = [
           '__trecis2019-A-test.earthquakeBohol2013.csv',
           '__trecis2019-A-test.earthquakeCalifornia2014.csv',
           '__trecis2019-A-test.fireYMM2016.csv',
           '__trecis2019-A-test.floodChoco2019.csv',
           '__trecis2019-A-test.hurricaneFlorence2018.csv',
           '__trecis2019-A-test.shootingDallas2017.csv'
]

df1 = []
df2 = []

for i in range(len(TrainDataDir)):
  df1.append( pd.read_csv(TrainDataDir[i]) )

for i in range(len(TestDataDir)):
  df2.append( pd.read_csv(TestDataDir[i]) )


import sys
original_stdout = sys.stdout

cc = 0
for i in range(len(df1)):
  arr = []
  columns = list(df1[i].columns)
  print(columns)
  for row in df1[i].itertuples(name=None):
    if lang_detection(row[2]) == 'en':
      if cc <= 10:
        cc += 1
        print(row[1], row[2])
      temp = [row[1], preprocess_pipeline(row[2])]
      for j in range(3, len(row)):
        temp.append(row[j])
      arr.append(temp)

  yoyo = 'p' + TrainDataDir[i]
  with open(yoyo, 'w') as output:
    sys.stdout = output
    for j in range(len(columns)):
      if j == len(columns)-1:
        print(columns[j], end='\n')
      else:
        print(columns[j], end='\t')
    for j in range(len(arr)):
      for k in range(len(arr[j])):
        if k == (len(arr[j]) - 1):
          print(arr[j][k], end='\n')
        else:
          print(arr[j][k], end='\t')
  sys.stdout = original_stdout
  
cc = 0
for i in range(len(df2)):
  arr = []
  columns = list(df2[i].columns)
  print(columns)
  for row in df2[i].itertuples(name=None):
    if cc <= 10:
      cc += 1
      print(row[1], row[2])
    temp = [row[1], preprocess_pipeline(row[2])]
    for j in range(3, len(row)):
      temp.append(row[j])
    arr.append(temp)
  

  yoyo = 'p' + TestDataDir[i]
  with open(yoyo, 'w') as output:
    sys.stdout = output
    for j in range(len(columns)):
      if j == len(columns)-1:
        print(columns[j], end='\n')
      else:
        print(columns[j], end='\t')
    for j in range(len(arr)):
      for k in range(len(arr[j])):
        if k == (len(arr[j]) - 1):
          print(arr[j][k], end='\n')
        else:
          print(arr[j][k], end='\t')
  sys.stdout = original_stdout