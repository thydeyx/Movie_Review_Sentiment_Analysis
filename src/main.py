# -*- coding:utf-8 -*-
#
#        Author : TangHanYi
#        E-mail : thydeyx@163.com
#   Create Date : 2018-09-12 18:09:08
# Last modified : 2018-09-12 18:09:11
#     File Name : main.py
#          Desc :

from process import Process

import keras
import os
from keras import backend as K
from keras.datasets import imdb
from keras import regularizers
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import Embedding, Flatten
from keras.layers import LSTM, TimeDistributed, Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, Callback
from keras.backend.tensorflow_backend import set_session
import numpy as np


class TrainVaildTensorBoard(TensorBoard):

    def __init__(self, log_dir='./logs', x_train=[], y_train=[], **kwargs):
        self.traing_log_dir = os.path.join(log_dir, 'training')
        super(TrainVaildTensorBoard, self).__init__(**kwargs)
        self.val_log_dir = os.path.join(log_dir, 'val')
        self.batch_count = -1
        self.t_loss = 0.0
        self.t_acc = 0.0

    def set_model(self, model):
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        self.train_writer = tf.summary.FileWriter(self.traing_log_dir)
        super(TrainVaildTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}
        print('\n', logs)
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        self.batch_count += 1
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            if 'loss' in name:
                self.val_writer.add_summary(summary, self.batch_count)
            else:
                self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        for name, value in logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            if 'loss' in name:
                self.train_writer.add_summary(summary, self.batch_count)
            else:
                self.train_writer.add_summary(summary, epoch)
        self.train_writer.flush()

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        val_data = self.validation_data
        try:
            batch = logs['batch']
            loss = logs['loss']
            acc = logs['acc']
            if batch % 10 == 0 and batch != 0:
                t_loss = self.t_loss / 10.0
                t_acc = self.t_acc / 10.0
                self.t_loss = 0.0
                self.t_acc = 0.0
                self.batch_count += 1

                """
                y_pred = tf.convert_to_tensor(self.model.predict(val_data[0]), np.float32)
                y_true = tf.convert_to_tensor(val_data[1], np.float32)
                val_loss = K.categorical_crossentropy(y_true, y_pred)
                #print(np.asarray(val_loss, np.float32))
                loss_list = self.sess.run(val_loss)
                val_loss = np.sum(loss_list) / len(loss_list)
                print(' - val_loss', val_loss)
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = val_loss
                summary_value.tag = 'loss'
                self.val_writer.add_summary(summary, self.batch_count)
                self.val_writer.flush()
                #batch_logs = {'loss':t_loss, 'acc':t_acc}
                """
                batch_logs = {'loss': t_loss}

                for name, value in batch_logs.items():
                    summary = tf.Summary()
                    summary_value = summary.value.add()
                    summary_value.simple_value = value
                    summary_value.tag = name
                    self.train_writer.add_summary(summary, self.batch_count)
                self.train_writer.flush()
            else:
                self.t_loss += loss
                self.t_acc += acc

        except Exception as e:
            print(e)

    def on_train_end(self, logs=None):
        # super(TrainVaildTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
        self.train_writer.close()


class Solution:

    def __init__(self):
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        self.log_filepath = '../data/keras_log'
        es = EarlyStopping(monitor='val_acc', patience=5)
        # tb_cb = keras.callbacks.TensorBoard(log_dir=self.log_filepath, histogram_freq=0)
        tb_cb = TrainVaildTensorBoard(log_dir=self.log_filepath, histogram_freq=0)
        # tb_sb = TrainValidTensorBoardCallback()
        # self.clkb = [es, tb_cb]
        self.clkb = [tb_cb]

    def convert_data(self, train_data):

        train_num = len(train_data)
        train_text = []

        for i in range(train_num):
            train_text.append(train_data[i][0])

        token = text.Tokenizer(lower=True, split=" ", char_level=False)
        token.fit_on_texts(train_text)
        process_train_data = token.texts_to_sequences(train_text)
        for i in range(train_data):
            process_train_data[i] = (process_train_data[i], train_data[i][0])

        return process_train_data

    def read_data(self):

        classes = 5

        """
        (X_train, y_train), (X_test, y_test) = imdb.load_data(path='imdb.npz', num_words=None, skip_top=0, maxlen=None,
                                                              seed=113, start_char=0, oov_char=1, index_from=2)
        self.feature_len = 0
        self.vocab_size = 0
        for i in range(len(X_train)):
            self.feature_len = max(self.feature_len, len(X_train[i]))
            self.vocab_size = max(self.vocab_size, max(X_train[i]))
        self.feature_len = 200
        print('feature length:', self.feature_len)
        print('vocab size:', self.vocab_size)

        self.xtrain = pad_sequences(X_train, maxlen=self.feature_len, padding='post')
        # self.ytrain = y_train
        self.ytrain = []
        for i in range(len(y_train)):
            tmp = [0 for i in range(classes)]
            tmp[y_train[i]] = 1
            self.ytrain.append(tmp[:])
        self.xtest = pad_sequences(X_test, maxlen=self.feature_len, padding='post')
        # self.ytest = y_test
        self.ytest = []
        for i in range(len(y_test)):
            tmp = [0 for i in range(classes)]
            tmp[y_test[i]] = 1
            self.ytest.append(tmp[:])
        self.ytrain = np.array(self.ytrain)
        self.xtrain = np.concatenate((self.xtrain, self.xtest), axis=0)
        self.ytrain = np.concatenate((self.ytrain, self.ytest), axis=0)
        print(self.xtrain.shape)
        print(self.ytrain.shape)
        """

    def model(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size, output_dim=128))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(128))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        # parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model = multi_gpu_model(model, gpus=2)
        # parallel_model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        parallel_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # parallel_model.fit(self.xtrain, self.ytrain, batch_size=128, epochs=20, validation_data=(self.xtest, self.ytest), callbacks=self.clkb, verbose=1)
        parallel_model.fit(self.xtrain, self.ytrain, batch_size=1024, epochs=10,
                           validation_data=(self.xtest, self.ytest))
        score = parallel_model.evaluate(self.xtest, self.ytest, batch_size=32)
        print(score)

    def function_model(self):
        vector_input = Input(shape=(200,), dtype='int32', name='vector_input')
        embedding = Embedding(self.vocab_size, output_dim=64)(vector_input)
        lstm = Bidirectional(LSTM(256, return_sequences=True), merge_mode='concat')(embedding)
        dropout1 = Dropout(0.5)(lstm)
        lstm1 = Bidirectional(LSTM(1024, kernel_regularizer=regularizers.l2(0.01)))(dropout1)
        # lstm = LSTM(1024)(embedding)
        dropout = Dropout(0.5)(lstm1)
        # dropout = Flatten()(dropout)
        output = Dense(2, kernel_regularizer=regularizers.l2(0.01), activation='softmax')(dropout)
        parallel_model = Model(inputs=vector_input, outputs=output)
        # print(parallel_model.predict(self.xtrain[:10]).shape)
        parallel_model = multi_gpu_model(parallel_model, gpus=2)
        # parallel_model = multi_gpu_model(parallel_model, gpus=4)
        # parallel_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # parallel_model.fit(self.xtrain, self.ytrain, batch_size=128, epochs=20, validation_data=(self.xtest, self.ytest), callbacks=self.clkb, verbose=1)
        # parallel_model.fit(self.xtrain, self.ytrain, batch_size=256, epochs=10, validation_data=(self.xtest, self.ytest), callbacks=self.clkb)
        parallel_model.fit(self.xtrain, self.ytrain, batch_size=256, epochs=50, validation_split=0.01,
                           callbacks=self.clkb, verbose=1)
        # parallel_model.fit(self.xtrain, self.ytrain, batch_size=512, epochs=30, validation_split=0.2, callbacks=self.clkb, verbose=1)
        score = parallel_model.evaluate(self.xtrain[25000:], self.ytrain[25000:], batch_size=32)
        print(score)

    def run(self):
        # self.read_data()
        # self.model()
        self.function_model()

def run():
    pro = Process()
    s = Solution()
    train_data = pro.readFile('../data/train.tsv')
    test_data = pro.readFile('../data/test.tsv')
    s.convert_data(train_data)


if __name__ == "__main__":
    run()
