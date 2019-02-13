# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 08:38:14 2018
Examines which tokens in the Quora dataset pose the most significance. To do this:
1) Performs Recursive Feature Elimination on a simple Linear SVM of Bag of Words model.
2) Finds a) the most common words in the insincere datapool but not common in the sincere and 
b) vice versa
@author: Connor Favreau
"""

import os
import re
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD

df = pd.read_csv('train.csv')


def review_sentence(sentence):
    """
    SPECIAL NOTE:
    Could mess with x5 if wanting to keep each definition corresponding to the same word separate
    """
    x = sentence.lower()
    x2 = re.sub('\$[0-9]+', 'money', x)
    x3 = re.sub('1[0-9]{3}[^\d]', 'year', x2)
    x4 = re.sub('[0-9]+', 'number', x3)
    x5 = re.sub('[^\w\s]', '', x4)        
    x6 = re.sub("\'s", '', x5)
    x7 = re.sub("\'|\`", '', x6)
#    x8 = re.sub("(?<=[\s])[a-zA-Z]{1}(?=[\n])", "", x7)
    words = x7
    words = words
    return words

df['question_text'] = df['question_text'].apply(review_sentence)

bow = CountVectorizer(lowercase=True,stop_words='english',analyzer='word', max_df=0.9, min_df=0.0001)

df = shuffle(df)
vectorized = bow.fit_transform(df['question_text'])

X_train = vectorized[0:1000000]
y_train = df['target'][0:1000000]
X_test = vectorized[1000000:]
y_test = df['target'][1000000:]

from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE

#Recursive feature eliminatation to determine most important words
svm = LinearSVC()
rfe = RFE(svm,n_features_to_select=100)
rfe.fit(X_train,y_train)
rfe.score(X_test,y_test)
from sklearn.metrics import precision_recall_fscore_support

#forest.feature_importances_




####### DO THROUGH A COUNTER ##############  (really don't do)
from nltk.tokenize import word_tokenize
from collections import Counter

sincere = Counter()
insincere = Counter()

import csv
with open('train.csv',encoding='utf-8') as train:
    readtrain = csv.reader(train, delimiter=',')
    next(readtrain)
    for line in readtrain:
        processed = word_tokenize(review_sentence(line[1]))
        tokens = Counter(processed)
        if line[2] == '1':
            insincere += tokens
        else:
            sincere += tokens
            
total = sincere + insincere

#Most common words in both sincere and insincere datasets
sin = set([i[0] for i in sincere.most_common(3000)])
insin = set([i[0] for i in insincere.most_common(3000)])

#Most common words in the insincere but not sincere dataset
hotinsinwords = insin.difference(sin)
hotsinwords = sin.difference(insin)

file = open('hotinsincerequorawords2.csv', 'w')
datafile = csv.writer(file,delimiter=',')


for word in hotinsinwords:
    hotwords = [word,insincere[word]/total[word],sincere[word]/total[word]]
    print(hotwords)
    datafile.writerow(hotwords)
file.close()    

file = open('hotsincerequorawords2.csv', 'w')
datafile = csv.writer(file,delimiter=',')


for word in hotsinwords:
    hotwords = [word,insincere[word]/total[word],sincere[word]/total[word]]
    print(hotwords)
    datafile.writerow(hotwords)
      
file.close()
#############################################


    

    
