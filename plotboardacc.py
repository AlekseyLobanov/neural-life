#!/usr/bin/python
# -*- coding: utf-8 -*- 

__author__ = "Aleksey Lobanov"
__copyright__ = "Copyright 2016, Aleksey Lobanov"
__credits__ = ["Aleksey Lobanov"]
__license__ = "MIT"
__maintainer__ = "Aleksey Lobanov"
__email__ = "i@likemath.ru"

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches

from neurallife import loadKeras, generateData, getAccuracies, META_PARAMETERS


if __name__ == "__main__":
    plt.title("Neural Life")

    plt.xscale("log")
    plt.yscale("log")
    
    plt.xlabel("Train size")
    plt.ylabel("Board accuracy")

    plt_patches = []
    for meta_ind,N in enumerate(sorted(META_PARAMETERS.keys())):
        points_x = []
        points_y = []
        
        X_test, Y_test = generateData(N, 100000)
        
        for data_size in META_PARAMETERS[N]: 
            nn = loadKeras("models/model_{}_{}".format(N, int(data_size * 0.5)))
        
            cellAcc, boardAcc = getAccuracies(nn, X_test, Y_test)
            
            points_x.append(data_size * 0.5)
            points_y.append(boardAcc)
            
        plt.plot(points_x, points_y, "o", linestyle="-", color=cm.ocean(meta_ind/len(META_PARAMETERS)))
        plt_patches.append(mpatches.Patch(color=cm.ocean(meta_ind/len(META_PARAMETERS)), label="N={}".format(N)))

    plt.legend(handles=plt_patches, loc=2)
    plt.savefig("graphics-board.svg")
