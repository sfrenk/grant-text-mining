#/usr/bin/env python3

# This script was heavily inspired by a Medium article written by Illia Polosukhin:
# https://medium.com/@ilblackdragon/tensorflow-text-classification-615198df9231

###############################################################################
# todo list

# should I scramble the years during training?

##############################################################################

from __future__ import print_function
import re
import random
import numpy as np
from datetime import datetime
import data_setup
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.utils import to_categorical


start = datetime.now()

###############################################################################
# PARAMETERS

years = ['1986', '2016']
max_doc_length = 600
embedding_size = 10
train_sample_size = 8000
validation_sample_size = 2000 
test_sample_size = 2000

###############################################################################
# HYPERPARAMETERS
max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
epochs = 1

# READING IN DATA

print(str(datetime.now()) +  ": Loading data...")

train_abstracts = []
validation_abstracts = []
test_abstracts = []

train_labels = []
validation_labels = []
test_labels = []

# encode class values as integers

for year in years:    

    f = open("data/" + year + ".csv")
    for i in range(int(train_sample_size/len(years))):
        train_abstracts.append(f.readline().strip())
        train_labels.append(year)

    for i in range(int(validation_sample_size/len(years))):
        validation_abstracts.append(f.readline().strip())
        validation_labels.append(year)

    for i in range(int(test_sample_size/len(years))):
        test_abstracts.append(f.readline().strip())
        test_labels.append(year)

    f.close()

print(len(train_abstracts), 'train sequences')
print(len(validation_abstracts), 'validation sequences')
print(len(test_abstracts), 'test sequences')

# Encode labels
year_ints = dict(zip(years, [ x for x in range(len(years))]))

train_int = [ year_ints[x] for x in train_labels ]
y_train = to_categorical(train_int, len(years))

validation_int = [ year_ints[x] for x in validation_labels ]
y_validation = to_categorical(validation_int, len(years))

test_int = [ year_ints[x] for x in test_labels ]
y_test = to_categorical(test_int, len(years))

# estimate the size of the vocabulary
words = set(text_to_word_sequence(" ".join(train_abstracts + validation_abstracts + test_abstracts)))
vocab_size = len(words)
print("vocab size: " + str(vocab_size))
# integer encode the document
train_abstracts = [ one_hot(x, vocab_size) for x in train_abstracts ]
validation_abstracts = [ one_hot(x, vocab_size) for x in validation_abstracts ]
test_abstracts = [ one_hot(x, vocab_size) for x in test_abstracts ]
print(y_train[0:5])

###############################################################################
# TRAINING MODEL

print(str(datetime.now()) + ':Pad sequences (samples x time)')
x_train = sequence.pad_sequences(train_abstracts, maxlen=maxlen)
x_validation = sequence.pad_sequences(validation_abstracts, maxlen=maxlen)
x_test = sequence.pad_sequences(test_abstracts, maxlen=maxlen)

print('Build model...')
model = Sequential()
model.add(Embedding(vocab_size, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(years), activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(str(datetime.now()) + ':Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_validation, y_validation))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# Save model

model.save("model.h5")
