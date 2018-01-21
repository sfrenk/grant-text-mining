#https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import argparse

# Params
SEQ_LENGTH=50

HIDDEN_DIM=500
LAYER_NUM=2
BATCH_SIZE=50
GENERATE_LENGTH=500

nb_epoch = 3
#################### Data Pre-processing #######################

# Open the data
data = open(DATA_DIR, 'r').read()

# Determine vocab (all possible chars)
chars = list(set(data))
VOCAB_SIZE = len(chars)

# Create dictionaries for encoding/decoding the vocab
ix_to_char = {ix:char for ix, char in enumerate(chars)}
char_to_ix = {char:ix for ix, char in enumerate(chars)}

# Create input
X = np.zeros((len(data)/SEQ_LENGTH, SEQ_LENGTH, VOCAB_SIZE))
y = np.zeros((len(data)/SEQ_LENGTH, SEQ_LENGTH, VOCAB_SIZE))

for i in range(0, len(data)/SEQ_LENGTH):

	# Select sequence of chars
    X_sequence = data[i*SEQ_LENGTH:(i+1)*SEQ_LENGTH]
    
    # Encode sequence
    X_sequence_ix = [char_to_ix[value] for value in X_sequence]
    
    # Create empty matrix, which will be populated to represent the encoded sequence as a two dimensional object
    input_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
    for j in range(SEQ_LENGTH):
        input_sequence[j][X_sequence_ix[j]] = 1.
    X[i] = input_sequence

    # Create output object. This is simply the input sequence shifted by one (as you want the neural network to predict the next char in a sequence)
    y_sequence = data[i*SEQ_LENGTH+1:(i+1)*SEQ_LENGTH+1]
    y_sequence_ix = [char_to_ix[value] for value in y_sequence]
    target_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
    for j in range(SEQ_LENGTH):
        target_sequence[j][y_sequence_ix[j]] = 1.
    y[i] = target_sequence

################## Model ###########################

# Create model
model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
for i in range(LAYER_NUM - 1):
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

# Run model
while True:
    print('\n\n')
    model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, nb_epoch=1)
    nb_epoch += 1
    generate_text(model, GENERATE_LENGTH)
    if nb_epoch % 10 == 0:
        model.save_weights('checkpoint_{}_epoch_{}.hdf5'.format(HIDDEN_DIM, nb_epoch))

def generate_text(model, length):

	# Initialize sequence with random char
    ix = [np.random.randint(VOCAB_SIZE)]
    y_char = [ix_to_char[ix[-1]]]

    # Create empty matrix which will be populated with a sequence of characters generated by the model
    X = np.zeros((1, length, VOCAB_SIZE))
    for i in range(length):
        X[0, i, :][ix[-1]] = 1
        print(ix_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        y_char.append(ix_to_char[ix[-1]])
    return ('').join(y_char)