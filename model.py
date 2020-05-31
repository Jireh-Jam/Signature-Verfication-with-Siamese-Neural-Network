from __future__ import absolute_import
from __future__ import print_function
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten,MaxPooling2D, Dropout


def SiameseNetwork(input_shape):
    input = Input(shape=input_shape)
    x=Conv2D(filters=32, kernel_size = (3,3),activation='relu')(input)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    x=Conv2D(filters=64, kernel_size = (3,3),activation='relu')(x)
    x=MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.1)(x)
    x=Conv2D(filters=128, kernel_size = (3,3),activation='relu')(x)
    x=MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.1)(x)
    x=Flatten()(x)
    x=Dense(128,activation='sigmoid')(x)
    return Model(input,x)
