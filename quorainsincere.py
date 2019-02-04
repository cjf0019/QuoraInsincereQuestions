# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:19:57 2018

@author: InfiniteJest
"""
import os
import re
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD

df = pd.read_csv('train.csv')

vectorizer = TfidfVectorizer(max_df=0.9,min_df=0.0001, stop_words='english')
bow = CountVectorizer(max_features=65000)
    
def processtext(df,textcolumn):
    df['Processed'] = df[textcolumn].str.lower()
    df['Processed'] = df['Processed'].str.replace('?','')
    df['Processed'] = df['Processed'].str.replace('!','')
    df['Processed'] = df['Processed'].str.replace(',','')
    df['Processed'] = df['Processed'].str.replace('.','')
    df['Processed'] = df['Processed'].str.replace('(','')
    df['Processed'] = df['Processed'].str.replace(')','')
    df['Processed'] = df['Processed'].str.replace('?','')
    df['Processed'] = df['Processed'].str.replace('"','')
    df['Processed'] = df['Processed'].str.replace("'",'')
    return df

df = processtext(df,'question_text')
df = shuffle(df)

bags = bow.fit_transform(df['Processed'])
#tfidf = vectorizer.fit_transform(df['Processed'])

#Perform Latent semantic analysis either BoW or tf-idf matrix
svd = TruncatedSVD(n_components = 25)    
lsa = svd.fit_transform(bags)
print('Latent Semantic Analysis Done.')

#train, test = train_test_split((tfidf,df['target']))
X_train = lsa[0:1000000]
y_train = df['target'][0:1000000]
X_test = lsa[1000000:]
y_test = df['target'][1000000:]


#Use Random Forest on LSA results... had explored modifying the insincere weights for training
forest = RandomForestClassifier(class_weight={0:1,1:1000})
forest.fit(X_train,y_train)
forest.score(X_test,y_test)
forest.feature_importances_

#94.6% score

#SVM Attempt
from sklearn.svm import SVC
svm = SVC(class_weight={0:0.2,1:1})
svm.fit(X_train,y_train)
svm.score(X_test,y_test)


import pickle
decision_tree_pkl_filename = 'quora_decision_tree_classifier.pkl'
# Open the file to save as pkl file
decision_tree_model_pkl = open(decision_tree_pkl_filename, 'wb')
pickle.dump(forest, decision_tree_model_pkl)
# Close the pickle instances
decision_tree_model_pkl.close()


from sklearn.metrics import precision_recall_fscore_support
result = precision_recall_fscore_support(y_test,forest.predict(X_test))
print('Random Forest Results:',result)

y_test = list(y_test)
test = forest.predict(X_test)

#Get list of TP, FN's, FP's
testind = 0
tptext = []
fntext = []
fptext = []
for index, row in df[1000000:].iterrows():
    if row['target'] == 1:
        if test[testind] == 1:
            tptext.append(row['question_text'])
        else:
            fntext.append(row['question_text'])
    else:
        if test[testind] == 1:
            fptext.append(row['question_text'])
    testind += 1



tpindices = []
for i in range(len(test)):
    if test[i] == 1:
        if y_test[i] == 1:
            tpindices.append(i)
tpindices.sort()                  

FP = 0
TP = 0
for i in range(len(test)):
    if test[i] == 1:
        if y_test[i] == 1:
            TP += 1
        else:
            FP += 1
print(TP,'True positives')
print(FP,'False positives')  

#To get list of ngrams
bow = CountVectorizer(max_features=25000,ngram_range=(2,2))
bags = bow.fit_transform(df['Processed'])
bow.get_feature_names().index('how did')

indices = list(zip(*bags.nonzero())) #returns tuples of row,col of indices of n gram hits



