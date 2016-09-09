#!/usr/bin/python
# -*- coding: utf-8 -*- 

__author__ = "Aleksey Lobanov"
__copyright__ = "Copyright 2016, Aleksey Lobanov"
__credits__ = ["Aleksey Lobanov"]
__license__ = "MIT"
__maintainer__ = "Aleksey Lobanov"
__email__ = "i@likemath.ru"

import sys
from copy import deepcopy
from datetime import datetime
import logging

import numpy as np

from sklearn.cross_validation import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches


def initLogging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler('neurallife.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def neighbors(field, i, j, fsize):
    nsum = 0
    for l in range(1, 10):
        x = i - 1 + (l - 1) // 3
        y = j - 1 + (l + 2) % 3
        if -1 < x < fsize and -1 < y < fsize and field[x][y] == 1:
            nsum += 1
    nsum -= field[i][j]
    return nsum


def nextGen(field, fsize):
    tmp_field = deepcopy(field)
    for i in range(fsize):
        for j in range(fsize):
            neighb = neighbors(tmp_field, i, j, fsize)
            if field[i][j] == 1 and not (2 <= neighb <= 3):
                field[i][j] = 0
            elif field[i][j] == 0 and neighb == 3:
                field[i][j] = 1


def uniqueRows(data):
    uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
    return uniq.view(data.dtype).reshape(-1, data.shape[1])


def generateData(board_size, count=10**5):
    assert(2**(board_size**2) >= count)
    X = np.random.randint(2, size=(int(count*1.2),board_size*board_size))
    X = uniqueRows(X)[:count]
    Y = []
    for row in X:
        tmp_list = row.reshape((board_size,board_size)).tolist()
        nextGen(tmp_list, board_size)
        Y.append(tmp_list)
    return (X, np.asarray(Y).reshape(X.shape))
    

def loadKeras(path):
    model = keras.models.model_from_json(open(path + '.json').read())
    model.load_weights(path + '.h5')
    model.compile(loss='MSE', optimizer='nadam', metrics=[])

    logging.debug("Keras model loaded from {}".format(path))
    return model


def saveKeras(model, path):
    json_architecture = model.to_json()
    json_path = path + '.json'
    with open(json_path, 'w') as f:
        f.write(json_architecture)
    weights_path = path + '.h5'
    model.save_weights(weights_path, overwrite=True)
    

def getModel(n):
    nn = Sequential()
    nn.add(Dense(8*n**2, input_dim=n**2, init="normal", activation="sigmoid"))
    nn.add(Dense(5*n**2,init="normal", activation="sigmoid"))
    nn.add(Dense(n**2,init="normal", activation="sigmoid"))
    nn.compile(loss="MSE", optimizer="nadam", metrics=[])
    return nn
    
    
def getAccuracies(model,x_test,y_test):
    preds = model.predict(x_test)
    preds = np.rint(preds).astype("int")
    acc_square = 1.0 * (preds == y_test).sum() / y_test.size
    acc_boards = 0
    for pred, real in zip(preds, y_test):
        if (pred != real).sum() == 0:
            acc_boards += 1
    acc_boards = 1.0 * acc_boards / y_test.shape[0]
    return (acc_square, acc_boards)



META_PARAMETERS = {
    9:[409600],
}


if __name__ == "__main__":
    initLogging()

    plt.title("Neural Life")

    plt.xscale("log")

    plt.xlabel("Train size")
    plt.ylabel("Cell accuracy")

    plt_patches = []
    for meta_ind,N in enumerate(META_PARAMETERS):
        points_x = []
        points_y = []
        for data_size in META_PARAMETERS[N]:
            cur_time = datetime.now()        
                    
            X_train, X_test, Y_train, Y_test = train_test_split(
                *generateData(N, data_size),  # X and Y
                test_size=0.6,
                random_state=23
            )
            
            train_size = X_train.shape[0]
            
            nn = getModel(N)
            
            nn.fit(X_train, Y_train, nb_epoch=40, shuffle=False, verbose=1)
        
            cellAcc, boardAcc = getAccuracies(nn, X_test, Y_test)
            
            points_x.append(train_size)
            points_y.append(cellAcc)
            
            logging.info(("BIG model: for board {}x{} with train size={} cell accuracy is {:.5f}%, " +
            "board accuracy is {:.5f}% and delta with theoretical board accuracy " +
            "is {:.8f}%  it takes {}").format(
                N,
                N,
                train_size,
                100 * cellAcc,
                100 * boardAcc,
                100 * abs(boardAcc - cellAcc**(N**2)),
                datetime.now() - cur_time
            ))
            saveKeras(nn, "models/bigmodel_{}_{}".format(N,train_size))
        plt.plot(points_x, points_y, "o", linestyle="-", color=cm.ocean(meta_ind/len(META_PARAMETERS)))
        plt_patches.append(mpatches.Patch(color=cm.ocean(meta_ind/len(META_PARAMETERS)), label="N={}".format(N)))

    plt.legend(handles=plt_patches)
    plt.savefig("biggraphics.svg")
