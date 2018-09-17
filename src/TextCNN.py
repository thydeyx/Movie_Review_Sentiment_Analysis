# -*- coding:utf-8 -*-

import keras
from keras.initializers import RandomNormal, Constant
from keras.layers import Embedding, Input
from keras.layers import Conv2D, MaxPooling2D, Dense
from keras.layers import Concatenate, Flatten, Reshape
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping, TensorBoard
from keras.backend.tensorflow_backend import set_session
import keras.backend as K
import tensorflow as tf
import numpy as np
import os

class TrainVaildTensorBoard(TensorBoard):

    def __init__(self, log_dir='./logs', x_train=[], y_train=[], **kwargs):
        self.traing_log_dir = os.path.join(log_dir, 'training')
        super(TrainVaildTensorBoard, self).__init__(**kwargs)
        self.val_log_dir = os.path.join(log_dir, 'val')
        self.batch_count = -1
        self.t_loss = 0.0
        self.t_acc = 0.0
        self.count = 0
        self.log_epoch = 10

    def set_model(self, model):
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        self.train_writer = tf.summary.FileWriter(self.traing_log_dir)
        super(TrainVaildTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}
        print('\n', logs)
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        #self.batch_count += 1
        #print('batch_count:', self.batch_count)
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            if 'loss' in name:
                pass
                #self.val_writer.add_summary(summary, self.batch_count)
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
                pass
                #self.train_writer.add_summary(summary, self.batch_count)
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
            #print('batch: ', batch)
            self.count += 1
            if self.count % self.log_epoch == 0 and self.count != 0:
                #t_loss = self.t_loss / float(self.count)
                t_loss = loss
                t_acc = self.t_acc / float(self.count)
                self.t_loss = 0.0
                self.t_acc = 0.0
                self.count = 0
                self.batch_count += 1

                y_pred = tf.convert_to_tensor(self.model.predict(val_data[0]), np.float32)
                y_true = tf.convert_to_tensor(val_data[1], np.float32)
                val_loss = K.categorical_crossentropy(y_true, y_pred)
                #print(np.asarray(val_loss, np.float32))
                loss_list = self.sess.run(val_loss)
                val_loss = np.sum(loss_list) / len(loss_list)
                print(' - val_loss', val_loss)
                #print('batch :', str(self.batch_count) + ' ' + str(t_loss) + ' ' + str(val_loss))
                #print('--------')
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = val_loss
                summary_value.tag = 'loss'
                self.val_writer.add_summary(summary, self.batch_count)
                self.val_writer.flush()
                #batch_logs = {'loss':t_loss, 'acc':t_acc}
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

class TextCNN(object):

    def __init__(self, sentence_length, vocab_size, filter_size, embedding_size, conv_filter_size, classes):
        self.sentence_length = sentence_length
        self.vocab_size = vocab_size
        self.conv_filter_size = conv_filter_size
        self.embedding_size = embedding_size
        self.filter_sizes = [int(x) for x in filter_size.split(',')]
        self.classes = classes
        self.batch_size = 32

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        self.log_filepath = '../data/keras_log'
        es = EarlyStopping(monitor='val_acc', patience=5)
        tb_cb = TrainVaildTensorBoard(log_dir=self.log_filepath, histogram_freq=0)
        self.clkb = [tb_cb]

    def model(self):

        vector_input = Input(shape=(self.sentence_length,), dtype='int32', name='vector_input')
        embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_size,
                              embeddings_initializer=RandomNormal(mean=0.0, stddev=1.0, seed=None))(vector_input)
        embedding_expand = Reshape(target_shape=(self.sentence_length, self.embedding_size, 1))(embedding)
        #embedding_expand = K.expand_dims(embedding, axis=-1)
        print(embedding_expand)
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            conv = Conv2D(filters=self.conv_filter_size, kernel_size=[filter_size, self.embedding_size], padding='valid',
                       strides=[1, 1], data_format='channels_last', use_bias=True, activation='relu')(embedding_expand)
            #print(conv.shape)
            max_pooling = MaxPooling2D(pool_size=[self.sentence_length - filter_size + 1, 1], strides=[1, 1],
                                   padding='valid', data_format='channels_last')(conv)
            #print('Pooling:', max_pooling.shape)
            pooled_outputs.append(max_pooling)

        num_filters_total = self.conv_filter_size * len(self.filter_sizes)
        pooled = Concatenate(axis=-1)(pooled_outputs)
        #print(pooled)
        #flatten = Flatten(data_format='channels_last')(pooled)
        pool_flat = Reshape(target_shape=(num_filters_total, ))(pooled)
        #print(pool_flat)
        output = Dense(self.classes, activation='softmax', use_bias=True, bias_initializer= Constant(value=0.1),
                       kernel_initializer=RandomNormal(mean=0.0, stddev=1.0, seed=None))(pool_flat)
        #print(output)
        return Model(inputs=vector_input, outputs=output)


    def train(self, x_train, y_train):

        self.x_train = x_train
        self.y_train = y_train
        parallel_model = self.model()
        #parallel_model = multi_gpu_model(parallel_model, gpus=2)
        parallel_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        parallel_model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=50, validation_split=0.1,
                           callbacks=self.clkb, verbose=1)