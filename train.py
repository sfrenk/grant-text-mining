#/usr/bin/env python3

# Pr

# This script was heavily inspired by a Medium article written by Illia Polosukhin:
# https://medium.com/@ilblackdragon/tensorflow-text-classification-615198df9231

###############################################################################
# todo list

# should I scramble the years during training?

##############################################################################

#from process_text import process_text
#import tflearn
import re
import random
import pandas as pd
import numpy as np
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn import metrics
from datetime import datetime
#from tensorflow.contrib.layers.python.layers import encoders
import data_setup

learn = tf.contrib.learn

start = datetime.now()

###############################################################################
# PARAMETERS

years = ['1986', '2016']
max_doc_length = 600
embedding_size = 10
train_sample_size = 10000
test_sample_size = 1000

###############################################################################
# READING IN DATA

print(str(datetime.now()) +  ": Reading data...")

data_set = data_setup.dataSet(years)
data_set.load_train(train_sample_size)
data_set.load_test(test_sample_size)

train = data_set.dump_data("train")
test = data_set.dump_data("test")

print(train.head(100))

#exit(0)
###############################################################################
# TRAINING MODEL

# Set up the VocabularyProcessor object
# This maps every word to an index number and converts every document to a
# vector of indices of length(max_doc_length). Documents shorter than 
# max_doc_length are padded while documents longer than max_doc_length
# are clipped

print(str(datetime.now()) +  ": Preprocessing...")

vocab_processor = learn.preprocessing.VocabularyProcessor(max_doc_length)

train_rec = np.array(list(vocab_processor.fit_transform(train.record))) 
test_rec = np.array(list(vocab_processor.transform(test.record)))

print(train_rec[1])

n_words = len(vocab_processor.vocabulary_)

print('Total words: %d' % n_words)

def rnn_model(features, target):  
  """RNN model to predict from sequence of words to a class."""  
  # Convert indexes of words into embeddings.  
  # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and
  # then maps word indexes of the sequence into [batch_size, 
  # sequence_length, EMBEDDING_SIZE].  
  word_vectors = tf.contrib.layers.embed_sequence(      
    features, vocab_size=n_words, embed_dim=embedding_size, scope='words')   
  # Split into list of embedding per word, while removing doc length
  # dim. word_list results to be a list of tensors [batch_size, 
  # EMBEDDING_SIZE].  
  word_list = tf.unstack(word_vectors, axis=1)
  # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
  cell = tf.contrib.rnn.GRUCell(embedding_size)   
  # Create an unrolled Recurrent Neural Networks to length of  
  # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each 
  # unit.
  _, encoding = tf.contrib.rnn.static_rnn(cell, word_list, dtype=tf.float32)   
  # Given encoding of RNN, take encoding of last step (e.g hidden 
  # size of the neural network of last step) and pass it as features 
  # to fully connected layer to output probabilities per class.  
  target = tf.one_hot(target, 2, 1, 0)  
  logits = tf.contrib.layers.fully_connected(
     encoding, 2, activation_fn=None)  
  loss = tf.contrib.losses.softmax_cross_entropy(logits, target)   
  # Create a training op.
  train_op = tf.contrib.layers.optimize_loss(      
     loss, tf.contrib.framework.get_global_step(),      
     optimizer='Adam', learning_rate=0.01, clip_gradients=1.0)   
  return (      
     {'class': tf.argmax(logits, 1), 
      'prob': tf.nn.softmax(logits)},      
     loss, train_op)

classifier = learn.SKCompat(learn.Estimator(model_fn=rnn_model))
# Train and predict

print(str(datetime.now()) +  ": Training model...")
classifier.fit(train_rec, train.label, steps=10000)

print(str(datetime.now()) +  ": Making predictions...")
y_predicted = classifier.predict(test_rec)

score = metrics.accuracy_score(test['label'].tolist(), y_predicted['class']) 
print('Accuracy: {0:f}'.format(score))
print(str(datetime.now()) +  ": Finished")
finish = datetime.now() - start

print('Run duration: ' + str(finish)) 

print(y_predicted['class'])
print(test['label'].tolist())
