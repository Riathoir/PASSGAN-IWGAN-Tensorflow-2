#  PassGAN_Final_Year_Project - Replication of PassGAN paper using Tensorflow 2 & Keras
#  Copyright (C) 2020 RachelaHorner
#
#  This file is part of PassGAN_Final_Year_Project (PFYP).
#
#  PFYP is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  PFYP is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with PFYP.  If not, see <http://www.gnu.org/licenses/>.

import tensorflow as tf
import keras

from resnet import ResBlock


class BuildGenerator(keras.Model):
    def __init__(self, layer_dim, seq_len):
        super(BuildGenerator, self).__init__()
        dim = layer_dim
        self.dim = layer_dim
        self.seq_len = seq_len

        self.fc1 = keras.layers.Dense(128, activation='linear', input_shape=(dim * seq_len,))
        self.block = keras.Sequential([
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
        ])
        self.conv1 = keras.layers.Conv1D(64, 32, 1, padding='valid')
        self.softmax = keras.layers.Softmax(axis=1)

    def call(self, noise, **kwargs):
        output = self.fc1(noise)
        output = tf.reshape(output, (-1, 2, 128))
        output = self.block(output)
        output = tf.reshape(output, [1, 32, 8])
        output = self.conv1(output)
        output = tf.transpose(output, [0, 2, 1])
        output = self.softmax(output)
        return tf.reshape(output, [2, 1, 32])
