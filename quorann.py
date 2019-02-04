# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 18:22:59 2018

@author: InfiniteJest
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
from sklearn.utils import shuffle
from sklearn.decomposition import PCA

class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras. The original code is from Afshine and Shervine Amidi
    at https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly and 
    was modified to fit pandas dataframes and text better"""
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
df = pd.read_csv('trainsmall.csv')
df = shuffle(df)
partition = {}
#partition['train'] = list(df[:800000]['qid'])
#partition['valid'] = list(df[800000:1000000]['qid'])
#partition['test'] = list(df[1000000:]['qid'])

partition['train'] = list(df[:120000]['qid'])
partition['valid'] = list(df[120000:1600000]['qid'])
partition['test'] = list(df[160000:]['qid'])

df.index = df['qid']
labels = df['target'].to_dict()

vocabulary = 27500
tokenizer = text.Tokenizer(num_words=vocabulary, lower=True,split=' ')
tokenizer.fit_on_texts(df['question_text'][partition['train']].values)
#vocabulary = len(tokenizer.word_index)
df['vectorized'] = tokenizer.texts_to_sequences(df['question_text'].values)
df['vectorized'] = sequence.pad_sequences(list(df['vectorized']),maxlen=75).tolist()

params = {'dim': np.shape(df['vectorized'][0]),
          'batch_size': 100,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': True}

num_steps = 75
training_generator = DataGenerator(partition['train'], labels, df, 'vectorized', vocabulary, **params)
validation_generator = DataGenerator(partition['valid'], labels, df, 'vectorized', vocabulary, **params)


### The following is an implementation of "Cyclic Learning Rates" proposed in "Cyclical Learning Rates for Training Neural Networks"
### by Leslie Smith, 2015. The code is from https://github.com/cjf0019/CLR/blob/master/clr_callback.py . 

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

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

###############################################

#### The following is a function version of what follows below, for purposes of 
#### more rapidly exploring the parameter space
class LSTMNetwork(Sequential):
    def __init__(self,vocabulary,embedsize,lstms,activation,dropout,num_steps):
        super().__init__()
        self.vocabulary = vocabulary
        self.embedsize = embedsize
        self.lstms = lstms
        self.activation = activation
        self.dropout = dropout
        self.num_steps = num_steps
        self.add(Embedding(self.vocabulary,self.embedsize, input_length=num_steps))
        for i in range(len(lstms)-1):
            self.add(LSTM(self.lstms[i], return_sequences=True))
        self.add(LSTM(self.lstms[-1], return_sequences=False))
        self.add(Dropout(self.dropout))
        self.add(Dense(1, activation='sigmoid'))
        optimizer = Adam()
        self.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return


vocabularies = [5000, 10000, 20000]
embedsizes = [100, 200, 300]
num_stepsz = [104,100,75,50,25]
lstmsz = [[200,100],[100,100],[50,50],[100,25]]
dropout = [0.5,0.4,0.3,0.2,0.1]
#################################################################

embed_size = 100
hidden_size = 100
use_dropout=True
model = Sequential()
model.add(Embedding(vocabulary, embed_size, input_length=num_steps))
model.add(Bidirectional(LSTM(hidden_size, return_sequences=True, activation='tanh', \
               kernel_initializer='Identity')))


model.add(LSTM(50, return_sequences=False, activation='tanh', \
               kernel_initializer='Identity'))
if use_dropout:
    model.add(Dropout(0.4))
#model.add(Dense(16, activation='relu', input_shape=(num_steps,hidden_size)))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(clipnorm=1.0)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['mean_squared_error'])

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False)


#Evaluate the model
#test_generator = DataGenerator(partition['test'], labels, df, 'vectorized', vocabulary, **params)

#model.evaluate_generator(test_generator)

from sklearn import metrics
#pred_noemb_val_y = model.predict_generator(test_generator, verbose=1)
pred_noemb_val_y = model.predict(np.array(list(df['vectorized'][partition['test']].as_matrix())))


#### Taken from https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, \
          metrics.f1_score(df['target'][partition['test']], (pred_noemb_val_y>thresh).astype(int))))

##################################################################################

dftest = df[160000:]
dftest['predict'] = pred_noemb_val_y
dftest.loc[dftest['predict'].idxmax()]
dfpred1 = dftest.loc[dftest['predict'] == 1]

#show whole text
pd.set_option('max_colwidth', 800)


def make_intermediate_model(model, layer_id):
    from keras.models import Model
    layers = [l for l in model.layers]
    
    x = model.layers[0].output
    for i in range(1, layer_id+1):
        x = layers[i](x)

    new_model = Model(inputs=layers[0].input, outputs=x)
    return new_model




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
    embed = Embedding(vocabulary, embed_size)(inputs)
    lstm_out = Bidirectional(LSTM(hidden_size, return_sequences=True, activation='tanh', \
                                  kernel_initializer='Identity'))(embed)
    gru_out = Bidirectional(GRU(hidden_size, return_sequences=True, activation='tanh', \
                                kernel_initializer='Identity'))(lstm_out)
    if use_dropout:
        gru_out = Dropout(0.3)(gru_out)
    attention_mul = attention_3d_block(gru_out,num_steps)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = keras.models.Model(inputs=[inputs], outputs=output)
    return model

def model_attention_applied_after_lstm(num_steps,embed_size,hidden_size, use_dropout=True):
    inputs = Input(shape=(num_steps,))
    embed = Embedding(vocabulary, embed_size)(inputs)
    lstm_out = Bidirectional(LSTM(hidden_size, return_sequences=True, activation='tanh', \
                                  kernel_initializer='Identity'))(embed)
    if use_dropout:
        lstm_out = Dropout(0.2)(lstm_out)
    attention_mul = attention_3d_block(lstm_out,num_steps)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = keras.models.Model(inputs=[inputs], outputs=output)
    return model



### ATTEMPT TO ADD TOPIC %'s to the LSTM

from KerasCustomLayers import LDALSTM
def model_attention_applied_after_lstm(num_steps,embed_size,hidden_size, num_topics, use_dropout=True):
    inputs = [Input(shape=(num_steps,)), Input(shape=(num_topics,))]
    embed = Embedding(vocabulary, embed_size)(inputs[0])
    lstm_out = LDALSTM(hidden_size, return_sequences=True, activation='tanh', \
                                  kernel_initializer='Identity')([embed,inputs[1]])
    if use_dropout:
        lstm_out = Dropout(0.2)(lstm_out)
    attention_mul = attention_3d_block(lstm_out,num_steps)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = keras.models.Model(inputs=[inputs], outputs=output)
    return model





embed_size = 100
hidden_size = 100
use_dropout=True
attnmodel = model_attention_applied_after_lstm(num_steps,embed_size,hidden_size)
attnmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mean_squared_error'])

attnmodel.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False)


from keras import backend as K

inp = attnmodel.input                                 # input placeholder
outputs = [layer.output for layer in attnmodel.layers[1:]]          # all layer outputs
functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions

# Testing
test = np.random.random((1,75))[np.newaxis,...]
layer_outs = [func([test, 1.]) for func in functors]
print(layer_outs)

shapes = [(1,75),(1,75),(1,75,100),(1,1,200),(1,1,200),(1,200,1),(1,200,75),(1,200,75),\
          (1,1,200),(1,75,200),(1,75,200),(1,1)]
for layer in attnmodel.layers[1:]:
    inp = layer.input
    functor = K.function([inp,K.learning_phase()], [out])
        





print(model.summary())
checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
num_epochs = 50
if args.run_opt == 1:
    model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
                        validation_data=valid_data_generator.generate(),
validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[checkpointer])
    
    
    
