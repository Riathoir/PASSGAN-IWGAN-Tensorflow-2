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


class ResBlock(keras.Model):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.res_block = keras.Sequential([
            keras.layers.ReLU(True),
            keras.layers.Conv1D(dim, dim, 5, padding='same'),
            keras.layers.ReLU(True),
            keras.layers.Conv1D(dim, dim, 5, padding='same'),
        ])

    def call(self, input, **kwargs):
        output = self.res_block(input)
        return input + (0.3 * output)
