import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import random

import matplotlib.pyplot as plt

class StegoDetectorExtended(Model):

    def __init__(self, shape, name='StegoDetectorExtended', dl_size = 128, drop_pcg=0.025, seed=123, **kwargs):
        super().__init__(name=name, **kwargs)

        self.input_layer = InputLayer(shape)
        self.seed = seed
        self.drop_pcg = drop_pcg
        self.dl_size = dl_size

        #block 0
        self.dl_0 = Dense(self.dl_size, activation="relu", kernel_initializer=glorot_normal(self.seed))
        self.bn_0 = BatchNormalization()
        self.dr_0 = Dropout(self.drop_pcg)

        self.dl_1 = Dense(self.dl_size, activation="relu", kernel_initializer=glorot_normal(self.seed))
        self.bn_1 = BatchNormalization()
        self.dr_1 = Dropout(self.drop_pcg)

        self.dl_2 = Dense(self.dl_size, activation="relu", kernel_initializer=glorot_normal(self.seed))
        self.bn_2 = BatchNormalization()
        self.dr_2 = Dropout(self.drop_pcg)

        self.output_layer = Dense(1, "sigmoid", kernel_initializer=glorot_normal(seed))


    def call(self, x):

        x1 = self.input_layer(x)

        #block 0
        y = self.dl_0(x1)
        y = self.bn_0(y)
        y = self.dr_0(y)

        # block 1
        y = self.dl_1(y)
        y = self.bn_1(y)
        y = self.dr_1(y)

        # block 2
        y = self.dl_2(y)
        y = self.bn_2(y)
        y = self.dr_2(y)

        y = self.output_layer(y)

        return y