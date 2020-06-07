from __future__ import absolute_import
from __future__ import print_function
from sklearn.utils import shuffle
import keras.backend as K
import cv2
import itertools
import numpy as np
import random
import configuration

def euclidean_distance(two_points):
    A, B = two_points
    sum_square = K.sum(K.square(A - B), axis=1, keepdims=True)
    res=K.maximum(sum_square, K.epsilon())
    res=K.sqrt(res)
    return res

def contrastive_loss(y_gt, y_pred):
    margn = 1
    square_pred = K.square(y_pred)
    res=K.maximum(margn - y_pred, 0)
    res = K.square(res)
    res=K.mean(y_gt * square_pred + (1 - y_gt) * res)
    return res

