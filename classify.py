#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
from sklearn import metrics
from sklearn import cross_validation
import pandas
import random

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import skflow
import cp 

class Classifier :

    def train(self,filename) :
        data = pandas.read_csv("train.txt",  header=None)
        X_data, y_data = data[1], data[0]

        ### Process vocabulary
        # The basic ideal is
        # 1. Convert each word to a EMBEDDING_SIZE-dimention vector. 
        # 2. Convert each document to a MAX_DOCUMENT_LENGTH word id list. 
        # 3. Convert each list to a feature vector.
        # 4. Train and classify using the feature vectors.
        MAX_DOCUMENT_LENGTH = 20

        vocab_processor = skflow.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
        X_data = np.array(list(vocab_processor.fit_transform(X_data)))

        n_words = len(vocab_processor.vocabulary_)
        print('Total words: %d' % n_words)

        ### Models

        EMBEDDING_SIZE = 1000

        def average_model(X, y):
            word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
                embedding_size=EMBEDDING_SIZE, name='words')
            features = tf.reduce_max(word_vectors, reduction_indices=1)
            return skflow.models.logistic_regression(features, y)

        def rnn_model(X, y):
            """Recurrent neural network model to predict from sequence of words
            to a class."""
            # Convert indexes of words into embeddings.
            # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
            # maps word indexes of the sequence into [batch_size, sequence_length,
            # EMBEDDING_SIZE].
            word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
                embedding_size=EMBEDDING_SIZE, name='words')
            # Split into list of embedding per word, while removing doc length dim.
            # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
            word_list = skflow.ops.split_squeeze(1, MAX_DOCUMENT_LENGTH, word_vectors)
            # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
            cell = rnn_cell.GRUCell(EMBEDDING_SIZE)
            # Create an unrolled Recurrent Neural Networks to length of
            # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
            _, encoding = rnn.rnn(cell, word_list, dtype=tf.float32)
            # Given encoding of RNN, take encoding of last step (e.g hidden size of the
            # neural network of last step) and pass it as features for logistic
            # regression over output classes.
            return skflow.models.logistic_regression(encoding, y)

        self.classifier =  skflow.TensorFlowEstimator(model_fn=rnn_model, n_classes=15, steps=1000, optimizer='Adam', learning_rate=0.01, continue_training=True)
        self.classifier.fit(X_data, y_data, logdir='/tmp/tf_examples/word_rnn')

    def predict(self):
        data = pandas.read_csv("testData.txt",  header=None)
        X_data= data[1]
        MAX_DOCUMENT_LENGTH = 20

        vocab_processor = skflow.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
        X_data = np.array(list(vocab_processor.fit_transform(X_data)))
        y_data = self.classifier.predict(X_data)
        for i in range(0,len(X_data)):
            print data[0][i]+','+str(y_data[i])+'\n'

    def crossValidation(self,filename):
        data = pandas.read_csv(filename,  header=None)
        X_data, y_data = data[1], data[0]
        MAX_DOCUMENT_LENGTH = 20

        vocab_processor = skflow.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
        X_data = np.array(list(vocab_processor.fit_transform(X_data)))

        n_words = len(vocab_processor.vocabulary_)

        ### Models

        EMBEDDING_SIZE = 1000

        def average_model(X, y):
            word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
                embedding_size=EMBEDDING_SIZE, name='words')
            features = tf.reduce_max(word_vectors, reduction_indices=1)
            return skflow.models.logistic_regression(features, y)

        def rnn_model(X, y):
            """Recurrent neural network model to predict from sequence of words
            to a class."""
            # Convert indexes of words into embeddings.
            # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
            # maps word indexes of the sequence into [batch_size, sequence_length,
            # EMBEDDING_SIZE].
            word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
                embedding_size=EMBEDDING_SIZE, name='words')
            # Split into list of embedding per word, while removing doc length dim.
            # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
            word_list = skflow.ops.split_squeeze(1, MAX_DOCUMENT_LENGTH, word_vectors)
            # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
            cell = rnn_cell.GRUCell(EMBEDDING_SIZE)
            # Create an unrolled Recurrent Neural Networks to length of
            # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
            _, encoding = rnn.rnn(cell, word_list, dtype=tf.float32)
            # Given encoding of RNN, take encoding of last step (e.g hidden size of the
            # neural network of last step) and pass it as features for logistic
            # regression over output classes.
            return skflow.models.logistic_regression(encoding, y)
        f = open('crossValidation.txt','w')
        while True:
            classifier = skflow.TensorFlowEstimator(model_fn=rnn_model, n_classes=15,
            steps=400, optimizer='Adam', learning_rate=0.01, continue_training=True)
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_data, y_data, test_size=0.4, random_state=random.randint(0,1000))
            classifier.fit(X_train, y_train, logdir='/tmp/tf_examples/word_rnn')
            score = metrics.accuracy_score(y_test, classifier.predict(X_test))
            print('Accuracy: {0:f}'.format(score))
            f.write('Accuracy: {0:f}'.format(score)+'\n')
        f.close()
#cha = cp.CP()
#cha.generateTestData()
#c = Classifier()
#c.crossValidation("train.txt")
#c.train()
#c.predict()