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

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense

import imageio  # for gifs

from PIL import Image, ImageDraw

from neurallife import nextGen


"""
# from original article
start_pos = [
    [0,0,0,0,1,0,1,0,0],
    [0,1,0,0,1,0,0,1,0],
    [0,1,1,0,1,1,0,1,0],
    [1,0,0,1,1,0,0,0,0],
    [0,1,1,1,0,1,0,1,0],
    [0,0,1,0,1,0,0,0,0],
    [0,0,1,1,0,0,1,0,0],
    [0,1,1,0,1,1,0,0,0],
    [0,0,0,0,0,0,0,0,0]
]
"""

# about 27 original positions
start_pos = [
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 1, 1, 1, 0],
    [1, 0, 1, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0]
]


"""
Code for good, long start positions:

def getBest(max_cnt):
    cur_max = 0
    cur_best = None
    for i in range(max_cnt):
        cur_field = np.random.randint(2, size=(9,9)).tolist()
        cur_cnt = getCnt(cur_field)
        if cur_cnt > cur_max:
            cur_max = getCnt(cur_field, 200)
            cur_best = cur_field
    return (cur_best,cur_max)


def getCnt(pos, max_cnt=100):
    cnt = 0
    ker_pred = pos
    cur_pos = pos
    while True:
        cnt += 1
        ker_pred = nn.predict(np.asarray(ker_pred).reshape((1,1,9,9)))
        ker_pred = np.rint(ker_pred).astype("int").reshape((9,9)).tolist()

        old_cur_pos = deepcopy(cur_pos)
        next_gen(cur_pos, len(cur_pos))
        if np.asarray(old_cur_pos).sum() == np.asarray(cur_pos).sum():
            break
        if ker_pred != cur_pos:
            break
        if cnt > max_cnt:
            return 0
    return cnt
"""

LINE_SIZE = 2
SQUARE_SIZE = 18
FRAME_COUNT = 30
FRAME_DELAY = 0.3  # in seconds


def loadKeras(path):
    model = keras.models.model_from_json(open(path + '.json').read())
    model.load_weights(path + '.h5')
    model.compile(loss='MSE', optimizer='nadam', metrics=[ ])
    return model


def imageFromList(l):
    global LINE_SIZE, SQUARE_SIZE
    height = LINE_SIZE * (len(l) + 1) + SQUARE_SIZE * len(l)  # =height
    width = LINE_SIZE * (len(l[0]) + 1) + SQUARE_SIZE * len(l[0])
    tmp_img = Image.new('RGB', (width, height), (0, 0, 0))
    pil_draw = ImageDraw.Draw(tmp_img)
    for y in range(len(l)):
        for x in range(len(l[0])):
            if l[y][x] == 0:
                pil_draw.rectangle((
                    x * (LINE_SIZE + SQUARE_SIZE) + LINE_SIZE,
                    y * (LINE_SIZE + SQUARE_SIZE) + LINE_SIZE,
                    (x + 1) * (LINE_SIZE + SQUARE_SIZE)-1,
                    (y + 1) * (LINE_SIZE + SQUARE_SIZE)-1,
                ), fill=(255, 255, 255))
    return tmp_img

N = 9  # board size

if __name__ == '__main__':
    nn = loadKeras(sys.argv[1])
    nn_frames = []
    real_frames = []
    ker_pred = cur_pos = start_pos
    
    for i in range(FRAME_COUNT):
        #ker_pred = cur_pos
        
        nn_frames.append(imageFromList(ker_pred))
        ker_pred = nn.predict(np.asarray(ker_pred).reshape((1, N**2)))
        ker_pred = np.rint(ker_pred).astype("int").reshape((N, N)).tolist()
        
        real_frames.append(imageFromList(cur_pos))
        old_pos = deepcopy(cur_pos)
        nextGen(cur_pos, len(start_pos))
        
        # because need some pause at end
        #if cur_pos == old_pos:
        #    break
    width, height = nn_frames[0].size
    
    imageio.mimsave(
        'gif_neural.gif',
        [np.asarray(img.getdata()).reshape((width, height, 3)) for img in nn_frames],
        fps=1/FRAME_DELAY
        )
    imageio.mimsave(
        'gif_real.gif',
        [np.asarray(img.getdata()).reshape((width, height, 3)) for img in real_frames],
        fps=1/FRAME_DELAY
        )

    

