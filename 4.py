import nltk
nltk.download()

import pandas
from nltk import FreqDist
from nltk.tokenize import wordpunct_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import math

porter = PorterStemmer()

import numpy as np

class MultinomialNaiveBayes:
  def __init__(self, nb_classes, nb_words, pseudocount):
    self.nb_classes = nb_classes
    self.nb_words = nb_words
    self.pseudocount = pseudocount
  
  def fit(self, X, Y):
   
    nb_examples = X.shape[0]
    
    self.priors = np.bincount(Y) / nb_examples
    print('Priors:')
    print(self.priors)

    occs = np.zeros((self.nb_classes, self.nb_words))
    for i in range(nb_examples):
      c = Y[i]
      for w in range(self.nb_words):
        cnt = X[i][w]
        occs[c][w] += float(cnt)
    print('Occurences:')
    print(occs)
    
    self.like = np.zeros((self.nb_classes, self.nb_words))
    for c in range(self.nb_classes):
      for w in range(self.nb_words):
        up = occs[c][w] + self.pseudocount
        down = np.sum(occs[c]) + self.nb_words*self.pseudocount
        self.like[c][w] = up / down
    print('Likelihoods:')
    print(self.like)
          
  def predict(self, bow):
    probs = np.zeros(self.nb_classes)
    for c in range(self.nb_classes):
      prob = np.log(self.priors[c])
      for w in range(self.nb_words):
        cnt = bow[w]
        prob += cnt * np.log(self.like[c][w])
      probs[c] = prob
    
    print('\"Probabilites\" for a test BoW (with log):')
    print(probs)
    prediction = np.argmax(probs)
    return prediction
    
  def predict_multiply(self, bow):
    probs = np.zeros(self.nb_classes)
    for c in range(self.nb_classes):
      prob = self.priors[c]
      for w in range(self.nb_words):
        cnt = bow[w]
        prob *= self.like[c][w] ** cnt
      probs[c] = prob
    print('\"Probabilities\" for a test BoW (without log):')
    print(probs)
    prediction = np.argmax(probs)
    return prediction


#corpus = pandas.read_csv("fake_news.csv")

def load_data():
  x = []
  y = []
  ty = []

  corpus = pandas.read_csv("fake_news.csv")

  x.append(corpus.text)
  ty.append(corpus.label)

  for i in range(0, len(ty[0])):
    y.append(ty[0][i])

  print(y)
  indexes = np.random.permutation(len(x))
  x = np.asarray(x)
  x = x[indexes]

  y = np.asarray(y)
  y = y[indexes]

  return x,y


from pandas.core.common import temp_setattr
import re
import html

clean_corpus = []

def clean(x):
  print('Cleaning the corpus...')
  stop_punc = set(stopwords.words('english')).union(set(punctuation))
  for doc in x:
    temp = wordpunct_tokenize(str(doc))
 
    words = [t.replace("'","") for t in temp]

    words_lower = [w.lower() for w in words]
    
    words_filtered = [w for w in words_lower if w not in stop_punc]
    
    words_stemmed = [porter.stem(w) for w in words_filtered]
    
    words_url = [html.unescape(w) for w in words_stemmed]
    re_url = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    words_url = [re.sub(re_url, '', w) for w in words_stemmed]

    words_spec = [re.sub(r'[^a-zA-Z0-9\s]', '', w) for w in words_url]

    clean_corpus.append(words_spec)
    print(clean_corpus)
    
  return clean_corpus

from collections import Counter

def create_vocabulary():
#count unique words
  def word_counter(text_col):
    count = Counter()
    for text in text_col:
      for word in str(text).split():
        count[word] += 1
    return count

  counter = word_counter(clean_corpus)

  mcs = counter.most_common(1000)
  print(mcs)

  final = []

  for mc in mcs:
    final.append((mc[0]).replace("\'",""))

  print(final)


  print('Creating the vocab...')
  vocab_set = set()
  for word in final:
    vocab_set.add(word)
  vocab = list(vocab_set)

  print('Vocab:', list(zip(vocab, range(len(vocab)))))
  print('Feature vector size: ', len(vocab))

  np.set_printoptions(precision=2, linewidth=200)
  return vocab

def numocc_score(word, doc):
    return doc.count(word)

def create_bow(doc,vocab):
  bow = np.zeros(len(vocab),dtype=np.float64)
  for word_idx in range(len(vocab)):
      word = vocab[word_idx]
      cnt = numocc_score(word,doc)
      bow[word_idx] = cnt
 
  return bow

def BoW_model(corpus,labels,vocab):
  print("Creating bow features...")
  X = np.zeros((len(clean_corpus),len(vocab)),dtype=np.float64)
  for doc_idx in range(len(clean_corpus)):
    doc = clean_corpus[doc_idx]
    X[doc_idx] = create_bow(doc,vocab)
  Y = np.zeros(len(clean_corpus),dtype=np.int32)
  for j in range(len(Y)):
    #Y[j] = y[j]
    #Y.append(y[j])
    Y = np.append(Y, y[j])
  print("finished bow")
  return X,Y

"""
def BoW_model(clean_corpus,vocab):
  def numocc_score(word, doc):
    return doc.count(word)

  print('Creating BOW features...')
  def create_bow():
    for score_fn in [numocc_score]:
      X = np.zeros((len(clean_corpus), len(vocab)), dtype=np.float32)
      for doc_idx in range(len(clean_corpus)):
        doc = clean_corpus[doc_idx]
        for word_idx in range(len(vocab)):
          word = vocab[word_idx]
          cnt = score_fn(word, doc)
          X[doc_idx][word_idx] = cnt
    print('X:')
    print(X)
    print()
    return X

  return create_bow()
"""

def five_most_freq(clean_corpus,y,class_type):
    dic = {}
    counter = 0

    for i in range(len(clean_corpus)):
        if y[i] == class_type:
            counter+=1
            for w in clean_corpus[i]:
                dic.setdefault(w,0)
                dic[w] +=1

    return sorted(dic,key=dic.get,reverse=True)[:5]


def LR(word,y,clean_corpus):

    porter = PorterStemmer()
    word = porter.stem(word.lower())
    p_count = 0
    n_count = 0

    for i in range(len(clean_corpus)):
        doc = clean_corpus[i]
        if y[i] == 1:
            p_count+=doc.count(word)
        elif y[i] == 0:
            n_count+=doc.count(word)

    result = 0

    if p_count > 9 and n_count > 9:
        result = p_count/n_count

    return result,word

def remove_duplicates(l):
    return list(dict.fromkeys(l))

x,y = load_data()
x_clean = clean(x)

limit = math.floor(len(x_clean) * 0.8)
train_corpus = x_clean[:limit]
train_labels = y[:limit]

test_corpus = x_clean[limit:]
test_labels = y[limit:]

vocabulary = create_vocabulary()

#for i in range(0, len(y)):
 #   y[i] = int(y[i])

#X = BoW_model(train_corpus,vocabulary)
X,Y = BoW_model(train_corpus,train_labels,vocabulary)

model = MultinomialNaiveBayes(nb_classes=2,nb_words=1000,pseudocount =1)
model.fit(X,Y)
correct_pred = 0
TP = 0
TN = 0
FP = 0
FN = 0

for i in range(len(test_corpus)):

    doc = test_corpus[i]
    label = test_labels[i]
    bow = create_bow(doc,vocabulary)
    prediction = model.predict_multiply(bow)

    if prediction == test_labels[i]:
        correct_pred+=1
    if prediction == 1 and test_labels[i] ==1:
        TP+=1
    elif prediction == 1 and test_labels[i] == 0:
        FP+=1
    elif prediction == 0 and test_labels[i] == 0:
        TN+=1
    elif prediction == 0 and test_labels[i] == 1:
        FN+=1

confusion_matrix = [[TN,FP],[FN,TP]]
acc = correct_pred/len(test_corpus)
print(acc)
print(confusion_matrix)

print("top 5 fake: ",five_most_freq(x_clean,y,0))
print("top 5 true: ",five_most_freq(x_clean,y,1))

lr_predictions = []

for i in range(len(train_corpus)):
    for word in train_corpus[i]:
        lr_predictions.append(LR(word,y,train_corpus))

lr_predictions.sort(key=lambda tup:tup[0])
lr_predictions = remove_duplicates(lr_predictions)
length = len(lr_predictions)

for p in lr_predictions:
    print([])