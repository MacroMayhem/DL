import os
from keras.preprocessing.text import  Tokenizer
import re
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


DATA_PATH = 'C:\\Users\\Aditya\\PycharmProjects\\DeepLearning\\data\\Books'

def sentence_read(filename):
    with open(filename) as f:
        text = f.read()
    return re.split(r' *[\.\?!][\'"\)\]]* *', text)

def load_data():
    # get the data paths
    train_path = os.path.join(DATA_PATH, "MarkTwain_HuckleberryFinn.txt")
    test_path = os.path.join(DATA_PATH, "MarkTwain_TomSawyer.txt")

    train_raw_1 =open(train_path,encoding="utf8").read().lower()
    train_raw_2  =open(test_path,encoding="utf8").read().lower()

    train_raw = train_raw_1

    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(train_raw)
    print(tokenizer.word_counts)
    return tokenizer, train_raw

tokenizer, train_raw = load_data()

train_raw = train_raw
n_chars = len(train_raw)

seq_length = 100

x_train  = []
y_train = []

for i in range(0,n_chars-seq_length):
    seq_in = train_raw[i:i+seq_length]
    seq_out= train_raw[i+seq_length]

    x_train.append([tokenizer.texts_to_sequences([seq_in])])
    y_train.append(tokenizer.texts_to_sequences(seq_out))

n_x_input = len(x_train)
x_train = np.reshape(np.asarray(x_train), (n_x_input, 1, seq_length))
x_train = x_train/len(tokenizer.word_counts)

y_train = np.reshape(np.asarray(y_train),(n_x_input,))
y_train = np_utils.to_categorical(y_train)

n_classes = y_train.shape[1]

inputs = Input(shape=x_train[0].shape)
lstm   = LSTM(256,name='lstm',return_sequences=True)(inputs)
dp_out = Dropout(0.2,name='dropout')(lstm)
lstm_2 = LSTM(256,name='lstm_2',return_sequences=True)(dp_out)
dp_out_2 = Dropout(0.2,name='dropout_2')(lstm_2)
lstm_3 = LSTM(256,name='lstm_3')(dp_out_2)
dp_out_3 = Dropout(0.2,name='dropout_3')(lstm_3)
d_1    = Dense(n_classes,activation='softmax')(dp_out_3)

model = Model(inputs=inputs,outputs=d_1)
model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

model_path = 'C:\\Users\\Aditya\\PycharmProjects\\DeepLearning\\models\\weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.summary()

model.fit(x_train, y_train, epochs=50, batch_size=64, callbacks=callbacks_list)
