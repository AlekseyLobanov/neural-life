#!/usr/bin/python
# -*- coding: utf-8 -*- 

__author__ = "Aleksey Lobanov"
__copyright__ = "Copyright 2016, Aleksey Lobanov"
__credits__ = ["Aleksey Lobanov"]
__license__ = "MIT"
__maintainer__ = "Aleksey Lobanov"
__email__ = "i@likemath.ru"

from datetime import datetime

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D

import numpy as np

from sklearn.cross_validation import train_test_split

from neurallife import generateData, getAccuracies, saveKeras


def getModel(n):
    nn = Sequential()
    nn.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(1, n, n)))
    nn.add(Activation('relu'))
    nn.add(Dropout(0.25))
    nn.add(Flatten())
    nn.add(Dense(4*n**2,init="normal", activation="sigmoid"))
    nn.add(Dropout(0.15))
    nn.add(Dense(n**2,init="normal", activation="sigmoid"))
    nn.compile(loss="MSE", optimizer="nadam", metrics=[])
    return nn

N = 9  # board size

if __name__ == "__main__":

    X_train, X_test, Y_train, Y_test = train_test_split(
        *generateData(N, 2*10**5),
        test_size=0.5,
        random_state=23
    )
    X_train = X_train.reshape((X_train.shape[0], 1, N, N))
    X_test = X_test.reshape((X_test.shape[0], 1, N, N))
    
    nn = getModel(N)
    
    cur_time = datetime.now()
    
    nn.fit(X_train, Y_train, nb_epoch=20, shuffle=False, verbose=1)
    
    cellAcc, boardAcc = getAccuracies(nn, X_test, Y_test)
    
    print(("for board {}x{} with train size={} cell accuracy is {:.5f}%, " +
    "board accuracy is {:.5f}% and delta with theoretical board accuracy " +
    "is {:.8f}%  it takes {}").format(
        N,
        N,
        X_train.shape[0],
        100 * cellAcc,
        100 * boardAcc,
        100 * abs(boardAcc - cellAcc**(N**2)),
        datetime.now() - cur_time
    ))
    saveKeras(nn, "models/convolutional_{}_{}".format(N, X_train.shape[0]))
