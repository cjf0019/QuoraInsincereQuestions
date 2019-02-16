# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 18:22:59 2018
Neural network training for Kaggle's Quora Insincere questions competition.
Includes preprocessing, Keras models, training, and testing.
The models include a standard LSTM, a and a biLSTM with attention.
@author: Connor Favreau
"""
from keras.models import Sequential, load_model
from keras.layers import *
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import *
import keras
import numpy as np
import pandas as pd
import os
import re
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from keras.preprocessing import text, sequence   


# Processing functions from: https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings
# and https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing

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

#Data only available to participants in the Kaggle comptetition.
df = pd.read_csv('../input/train.csv')
df = shuffle(df)

 # lower
df["question_text"] = df["question_text"].apply(lambda x: x.lower())

    # Clean the text
df["question_text"] = df["question_text"].apply(lambda x: clean_text(x))
    
    # Clean numbers
df["question_text"] = df["question_text"].apply(lambda x: clean_numbers(x))
    
    # Clean spelings
df["question_text"] = df["question_text"].apply(lambda x: replace_typical_misspell(x))



class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras...
        Original code from Shervine Amidi: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        but modified to fit Pandas dataframes
        """
    def __init__(self, list_IDs, labels, df, dfcol, vocab, batch_size=32,
                 dim=(32,32,32), n_channels=1, n_classes=10, shuffle=True, predictonly=False):
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
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        if len([i for i in list(np.shape(X)) if i == 1]) >= 1:
            X = np.squeeze(X)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample... changed to take arrays from a column (self.dfcol) in a dataframe (self.df)
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
    
from keras.preprocessing import text, sequence   
df = pd.read_csv('../input/train.csv')
df = shuffle(df)

#df_in = df[df['target'] == 1]
#train_df_in = df_in[:65000]
#test_df_in = df_in[:65000]
#train_df_sin = df[df['target'] == 0]
#test_df_sin = train_df_sin[500000:675000]
#train_df_sin = train_df_sin[:500000]
#train_df = pd.concat([train_df_in, train_df_sin])
#test_df = pd.concat([test_df_in, test_df_sin])
#train_df = shuffle(train_df)
#test_df = shuffle(test_df)

partition = {}
partition['train'] = list(df[:800000]['qid'])
partition['valid'] = list(df[800000:1000000]['qid'])
partition['test'] = list(df[1000000:]['qid'])

#partition['train'] = list(train_df[:400000]['qid'])
#partition['valid'] = list(train_df[400000:]['qid'])
#partition['test'] = list(train_df[400000:]['qid'])

#partition['train'] = list(df[:120000]['qid'])
#partition['valid'] = list(df[120000:1600000]['qid'])
#partition['test'] = list(df[160000:]['qid'])

df.index = df['qid']
labels = df['target'].to_dict()

#Tokenize sentences and process (one-hot and padding)
vocabulary = 75000
tokenizer = text.Tokenizer(num_words=vocabulary, lower=True,split=' ')
tokenizer.fit_on_texts(df['question_text'][partition['train']].values)
#vocabulary = len(tokenizer.word_index)
df['vectorized'] = tokenizer.texts_to_sequences(df['question_text'].values)
df['vectorized'] = sequence.pad_sequences(list(df['vectorized']),maxlen=75).tolist()

print("Tokenization complete.")

params = {'dim': np.shape(df['vectorized'][0]),
          'batch_size': 512,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': True}

num_steps = 75
training_generator = DataGenerator(partition['train'], labels, df, 'vectorized', vocabulary, **params)
validation_generator = DataGenerator(partition['valid'], labels, df, 'vectorized', vocabulary, **params)

### Embedding code taken from Sudalai Raj Kumar: https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(vocabulary, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= vocabulary: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

#Perform PCA to reduce dimensionality and runtime
pca = PCA(n_components=50)
embedding_matrix = pca.fit_transform(embedding_matrix)

###

### The following is an implementation of "Cyclic Learning Rates" proposed in "Cyclical Learning Rates for Training Neural Networks"
### by Leslie Smith, 2015. The code is from https://github.com/cjf0019/CLR/blob/master/clr_callback.py . 
### Inspiration for this method from Rahul Agarwhal: https://www.kaggle.com/mlwhiz/third-place-model-for-toxic-comments-in-pytorch

class CyclicLR(Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())


clr = CyclicLR(base_lr=0.001, max_lr=0.006, step_size=3125., mode='triangular')

print('CylicLR and Embeddings Set up.')

#The following attention model layers were taken from https://github.com/philipperemy/keras-attention-mechanism
def attention_3d_block(inputs,num_steps,SINGLE_ATTENTION_VECTOR=False):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, num_steps))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(num_steps, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = keras.layers.Multiply()([inputs, a_probs])
    return output_attention_mul


def model_attention_applied_after_lstm(num_steps,embed_size,hidden_size, use_dropout=True):
    inputs = Input(shape=(num_steps,))
    embed = Embedding(vocabulary, embed_size, weights=[embedding_matrix])(inputs)
    lstm_out = Bidirectional(LSTM(hidden_size, return_sequences=True, activation='tanh', \
                                  kernel_initializer='Identity'))(embed)
#    lstm_out = Bidirectional(LSTM(hidden_size, return_sequences=True, activation='tanh', \
#                                kernel_initializer='Identity'))(lstm_out)
    if use_dropout:
        lstm_out = Dropout(0.2)(lstm_out)
    attention_mul = attention_3d_block(lstm_out,num_steps)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = keras.models.Model(inputs=[inputs], outputs=output)
    return model

print("Training Model...")

embed_size = 50
hidden_size = 50
use_dropout=True
attnmodel = model_attention_applied_after_lstm(num_steps,embed_size,hidden_size)
attnmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mean_squared_error'])

attnmodel.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False, epochs=3, callbacks=[clr])

#test_df['vectorized'] = tokenizer.texts_to_sequences(test_df['question_text'].values)
#test_df['vectorized'] = sequence.pad_sequences(list(test_df['vectorized']),maxlen=75).tolist()
from sklearn import metrics
#pred_val_y = model.predict_generator(test_generator, verbose=1)

pred_val_y = attnmodel.predict(np.array(list(df['vectorized'][partition['test']].as_matrix())))

num = np.shape(pred_val_y)[0]

#Determine the best threshold from the test set
bestthresh = 0.1
bestf1 = 0
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    f1 = metrics.f1_score(df['target'][partition['test']][:num], (pred_val_y>thresh).astype(int))
    print("F1 score at threshold {0} is {1}".format(thresh, f1))
    if f1 > bestf1:
        bestf1 = f1
        bestthresh = thresh
print(bestthresh)


submitdata = pd.read_csv('../input/test.csv')
print(submitdata)
submitdata.index = submitdata['qid']
submitdata['vectorized'] = tokenizer.texts_to_sequences(submitdata['question_text'].values)
submitdata['vectorized'] = sequence.pad_sequences(list(submitdata['vectorized']),maxlen=75).tolist()
pred = attnmodel.predict(np.array(list(submitdata['vectorized'].as_matrix())))

pred[pred >= bestthresh] = 1
pred[pred < bestthresh] = 0
pred = pred.astype(int)
print(pred)

submission = pd.DataFrame(submitdata['qid'])
submission['prediction'] = pred
print(submission)
submission.to_csv("submission.csv", index=False)
