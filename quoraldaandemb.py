# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 18:13:21 2019

@author: InfiniteJest
"""

import gensim
import re
import os
import pandas as pd
from sklearn.utils import shuffle
from keras.preprocessing import text, sequence   


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", \
                       "'cause": "because", "could've": "could have", "couldn't": "could not", \
                       "didn't": "did not",  "doesn't": "does not", "don't": "do not", \
                       "hadn't": "had not", "hasn't": "has not", "haven't": "have not", \
                       "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", \
                       "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  \
                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", \
                       "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", \
                       "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have", \
                       "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", \
                       "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", \
                       "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", \
                       "might've": "might have","mightn't": "might not","mightn't've": "might not have", \
                       "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", \
                       "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", \
                       "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", \
                       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", \
                       "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", \
                       "she's": "she is", "should've": "should have", "shouldn't": "should not", \
                       "shouldn't've": "should not have", "so've": "so have","so's": "so as", \
                       "this's": "this is","that'd": "that would", "that'd've": "that would have", \
                       "that's": "that is", "there'd": "there would", "there'd've": "there would have", \
                       "there's": "there is", "here's": "here is","they'd": "they would", \
                       "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", \
                       "they're": "they are", "they've": "they have", "to've": "to have", \
                       "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", \
                       "we'll": "we will", "we'll've": "we will have", "we're": "we are", \
                       "we've": "we have", "weren't": "were not", "what'll": "what will", \
                       "what'll've": "what will have", "what're": "what are",  "what's": "what is", \
                       "what've": "what have", "when's": "when is", "when've": "when have", \
                       "where'd": "where did", "where's": "where is", "where've": "where have", \
                       "who'll": "who will", "who'll've": "who will have", "who's": "who is", \
                       "who've": "who have", "why's": "why is", "why've": "why have", \
                       "will've": "will have", "won't": "will not", "won't've": "will not have", \
                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", \
                       "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have", \
                       "y'all're": "you all are","y'all've": "you all have","you'd": "you would", \
                       "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", \
                       "you're": "you are", "you've": "you have" }

def known_contractions(embed):
    known = []
    for contract in contraction_mapping:
        if contract in embed:
            known.append(contract)
    return known

def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text

def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x

def unknown_punct(embed, punct):
    unknown = ''
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown

def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, ' {} '.format(p))
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text

def add_lower(embedding, vocab):
    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:
            embedding[word.lower()] = embedding[word]
            count += 1
    print("Added {} words to embedding".format(count))    
    
    
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
    x = str(x)
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, ' {} '.format(punct))
    return x


def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x

mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)
def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)


train_df = pd.read_csv('train.csv')
train_df = shuffle(train_df)

 # lower
train_df["question_text"] = train_df["question_text"].apply(lambda x: x.lower())

    # Clean the text
train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))
    
    # Clean numbers
train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_numbers(x))
    
    # Clean speelings
train_df["question_text"] = train_df["question_text"].apply(lambda x: replace_typical_misspell(x))


### Perform Latent Dirichlet Allocation on tf-idf matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
#bow = CountVectorizer(lowercase=True,stop_words='english',analyzer='word', max_df=0.9, min_df=0.0001)
tfidf = TfidfVectorizer(max_df=0.8,min_df=0.00001, max_features=27500, stop_words='english')

vectorized = tfidf.fit_transform(train_df['question_text'])
lda = LatentDirichletAllocation(n_components=3, random_state=0)
ldaresult = lda.fit_transform(vectorized) 

import pickle
#pickle.dump(lda, open('ldamodeltfidf3topics.pk', 'wb'))

with open('ldamodeltfidf3topics.pk', 'rb') as f:
    lda = pickle.load(f)


import numpy as np
topic_words = lda.components_
vocabulary = tfidf.get_feature_names()
n_top_words = 50
for i, topic_dist in enumerate(topic_words):
     topic_words = np.array(vocabulary)[np.argsort(topic_dist)][:-n_top_words:-1]
     print('Topic {}: {}'.format(i, ' '.join(topic_words)))

ldaresult = lda.transform(vectorized)
train_df['lda'] = list(ldaresult)

insincere = train_df[train_df['target'] == 1]
sincere = train_df[train_df['target'] == 0]






df = train_df[0:200000]
ldaresult = lda.transform(vectorized[0:200000])
df['lda'] = list(ldaresult)
from keras.preprocessing import text, sequence   
partition = {}

partition['train'] = list(df[:120000]['qid'])
partition['valid'] = list(df[120000:1600000]['qid'])
partition['test'] = list(df[160000:]['qid'])

df.index = df['qid']
labels = df['target'].to_dict()

vocabulary = 75000
tokenizer = text.Tokenizer(num_words=vocabulary, lower=True,split=' ')
tokenizer.fit_on_texts(df['question_text'][partition['train']].values)

df['vectorized'] = tokenizer.texts_to_sequences(df['question_text'].values)
df['vectorized'] = sequence.pad_sequences(list(df['vectorized']),maxlen=75).tolist()


"""
In what follows, a custom attention bi-directional LSTM model was created that incorporates
LDA topic weights into training. Three separate attention bi-directional LSTM models 
are trained concurrently, and, at the last step of the overall network, these 
networks are 


"""

import keras
class LDADataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, df, dfcol, vocab, batch_size=32,
                 dim=(32,32,32), n_channels=1, n_classes=10, num_topics = 1, shuffle=True, predictonly=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.df = df   #added to make applicable to dataframes
        self.dfcol = dfcol   #added to make applicable to dataframes
        self.vocab = vocab #convert integer list input into one-hot vectors
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_topics = num_topics
        self.shuffle = shuffle
        self.predictonly = predictonly
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        if isinstance(self.dfcol, list):
            X = [np.empty((self.batch_size, *self.dim, self.n_channels)), np.empty((self.batch_size, self.num_topics))]
            if len([i for i in list(np.shape(X[0])) if i == 1]) >= 1:
                X[0] = np.squeeze(X[0])

        else:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
            if len([i for i in list(np.shape(X)) if i == 1]) >= 1:
                X = np.squeeze(X)

        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample... changed to take arrays from a column (self.dfcol) in a dataframe (self.df)
            if isinstance(self.dfcol, list):
                for col in range(len(self.dfcol)):
                    X[col][i,] = self.df[self.dfcol[col]][ID]
            else:
                X[i,] = self.df[self.dfcol][ID]
#            X[i,] = self.df[self.dfcol][ID]

            # Store class
#            y[i] = keras.utils.to_categorical(self.labels[ID], num_classes=self.vocab)
            y[i] = self.labels[ID]

#        return keras.utils.to_categorical(X, num_classes=self.vocab), \
#                                          keras.utils.to_categorical(y, num_classes=self.n_classes)
        if self.predictonly:
            return X
        else:
            if self.n_classes > 1:
                return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
            else:
                return X, y

params = {'dim': np.shape(df['vectorized'][0]),
          'batch_size': 100,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': True}

num_steps = 75
training_generator = LDADataGenerator(partition['train'], labels, df, \
                                   ['vectorized','lda'], vocabulary, num_topics=3, **params)
validation_generator = LDADataGenerator(partition['valid'], labels, df, \
                                     ['vectorized','lda'], vocabulary, num_topics=3, **params)



from keras.models import Sequential, load_model
from keras.layers import *
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import *
from keras import backend as K

def attention_3d_block(inputs,num_steps,SINGLE_ATTENTION_VECTOR=False,tag='1'):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, num_steps))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(num_steps, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction'+tag)(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec'+tag)(a)
    output_attention_mul = keras.layers.Multiply()([inputs, a_probs])
    return output_attention_mul

def model_attention_applied_after_lstm(inputs,num_steps,embed_size,hidden_size, \
                                       use_dropout=True, tag='1'):
    embed = Embedding(vocabulary, embed_size)(inputs)
    lstm_out = Bidirectional(LSTM(hidden_size, return_sequences=True, activation='tanh', \
                                  kernel_initializer='Identity'))(embed)
    if use_dropout:
        lstm_out = Dropout(0.3)(lstm_out)
    attention_mul = attention_3d_block(lstm_out,num_steps,tag=tag)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    return output

def lda_attention(num_steps,embed_size,hidden_size, num_topics, use_dropout=True):
    inputs = [Input(shape=(num_steps,)), Input(shape=(num_topics,))]
    attns = []
    for topic in range(num_topics):
                attns.append(model_attention_applied_after_lstm(inputs[0],num_steps,embed_size,\
                                                             hidden_size,tag=str(topic)))
    stacked = keras.layers.concatenate(attns, axis=1)
#    stacked = Permute((2,1))(stacked)
    output = keras.layers.Dot(1)([stacked,inputs[1]])
    model = keras.models.Model(inputs=inputs, outputs=output)
    return model

num_steps = 75
embed_size = 100
hidden_size = 100

model = lda_attention(num_steps,embed_size,hidden_size,3)

optimizer = Adam(clipnorm=1.0)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['mean_squared_error'])

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False)

from sklearn import metrics
test_generator = LDADataGenerator(partition['test'], labels, df, \
                                     ['vectorized','lda'], vocabulary, num_topics=3, **params)

params = {'dim': np.shape(df['vectorized'][0]),
          'batch_size': 400,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': False}
pred_noemb_val_y = model.predict_generator(test_generator, verbose=1)

for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, \
          metrics.f1_score(df['target'][partition['test']], (pred_noemb_val_y>thresh).astype(int))))


pred_noemb_val_y = model.predict([list(df['vectorized'][partition['test']].as_matrix()),list(df['lda'][partition['test']])])
#BEST F1 WAS 0.08954 AT 0.1 THRESH!!!!

### 0.6363 ON LARGE DATASET
