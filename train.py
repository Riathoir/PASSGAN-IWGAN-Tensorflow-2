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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tempfile
from datetime import datetime
from functools import partial
from pathlib import Path

import tensorflow as tf
import keras
import tensorflow_datasets as tfds
from absl import flags
from tensorflow import random
from tensorflow.python.keras import metrics

from ops import d_loss_fn, wasserstein_loss
from discriminator import BuildDiscriminator
from generator import BuildGenerator
from utils import pbar

tf.config.experimental_run_functions_eagerly(False)
FLAGS = flags.FLAGS

"""
Wasserstein Generative Adversarial Network - Gradient Penalty main class

Parameters:
dataset: Name of selected Dataset contained within Tensorflow Database.
iterations: Number of iterations for training.
z_dim: Random vector noise size used in Gaussian Distribution.
epochs: Number of times dataset will be iterated through.
batch_size: Number of samples selected from the dataset per iteration.
seq_len: Maximum sequence length per sample.
layer_dim: Number of dimensions per layer.
n_critic: Critic updates per generator update.
vocab_size: Vocabulary size for selected dataset.
g_penalty: Gradient Penalty Weight.
total_num_examples: Total sample size of dataset, NOTE: this ignores the % taken and will return entire size.
encoder: ByteTextEncoder used to encode dataset features.
BuildGenerator: Generator model builder
BuildDiscriminator: Discriminator model builder
g_opt: Generator Adam Optimiser
d_opt: Discriminator Adam Optimiser
"""


class WGANGP:
    def __init__(self, dataset_info):
        self.dataset_name = FLAGS.dataset
        self.iterations = FLAGS.iterations
        self.checkpoints = FLAGS.checkpoints
        self.z_dim = FLAGS.z_size
        self.epochs = FLAGS.epochs
        self.batch_size = FLAGS.batch_size
        self.seq_len = FLAGS.seq_len
        self.layer_dim = FLAGS.layer_dim
        self.n_critic = FLAGS.n_critic
        self.vocab_size = FLAGS.vocab_size
        self.grad_penalty_weight = FLAGS.g_penalty
        self.total_passwords = dataset_info.splits.total_num_examples
        self.encoder = dataset_info.features['password'].encoder
        self.G = BuildGenerator(layer_dim=self.layer_dim, seq_len=self.seq_len)
        self.D = BuildDiscriminator(layer_dim=self.layer_dim, seq_len=self.seq_len)
        self.g_opt = keras.optimizers.Adam(learning_rate=FLAGS.g_lr, beta_1=0.5, beta_2=0.9)
        self.d_opt = keras.optimizers.Adam(learning_rate=FLAGS.d_lr, beta_1=0.5, beta_2=0.9)

    """
    Training steps for both
    Discriminator and Generator
    """

    def train(self, dataset):
        g_train_loss = metrics.Mean()
        d_train_loss = metrics.Mean()
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_directory = "./checkpoints/training_checkpoints"
        g_checkpoint_prefix = os.path.join(checkpoint_directory + "/generator", "ckpt")
        d_checkpoint_prefix = os.path.join(checkpoint_directory + "/discriminator", "ckpt")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        for epoch in tf.range(self.epochs):
            epoch = tf.cast(epoch, dtype=tf.int64, name=epoch)
            bar = pbar(self.total_passwords, self.batch_size, epoch, self.epochs)
            for iteration, batch in zip(range(self.iterations), dataset):
                for _ in tf.range(
                        self.n_critic):
                    self.text = batch['password']
                    real = tf.reshape(tf.dtypes.cast(self.text, tf.float32), [2, 1, 32])
                    self.train_d(real)
                    d_loss = self.train_d(real)
                    d_train_loss(d_loss)

                g_loss = self.train_g()
                g_train_loss(g_loss)
                self.train_g()

                """
                Tensorboard tracking calls
                files generated sent to
                /logs/gradient_tape/
                """
                with train_summary_writer.as_default():
                    tf.summary.scalar('Generator', g_train_loss.result(), step=epoch)
                    tf.summary.scalar('Accuracy', d_train_loss.result(), step=epoch)

                bar.postfix['g_loss'] = f'{g_train_loss.result():6.3f}'
                bar.postfix['d_loss'] = f'{d_train_loss.result():6.3f}'
                bar.update(self.batch_size)

                if iteration % self.checkpoints == 0 and iteration > 0:
                    generator_checkpoint = tf.train.Checkpoint(optimizer=self.g_opt, model=self.G)
                    generator_checkpoint.save(file_prefix=g_checkpoint_prefix)

                    discriminator_checkpoint = tf.train.Checkpoint(optimizer=self.d_opt, model=self.D)
                    discriminator_checkpoint.save(file_prefix=d_checkpoint_prefix)

            self.G.summary()
            self.D.summary()

            """
            Tensorflow model save
            files located at: /models/generator or /models/discriminator
            """
            tf.saved_model.save(self.G, './models/generator/' + self.dataset_name + current_time)
            tf.saved_model.save(self.D, './models/discriminator/' + self.dataset_name + current_time)

            g_train_loss.reset_states()
            d_train_loss.reset_states()

            bar.close()
            del bar

    """
    Generator training steps using Gradient Tape Graphing
    """

    @tf.function
    def train_g(self):
        z = tf.random.normal([2, 1, 32], dtype=tf.dtypes.float32)
        with tf.GradientTape() as t:
            t.watch(z)
            x_fake = self.G(z, training=True)
            fake_logits = self.D(x_fake, training=True)
            loss = wasserstein_loss(fake_logits)
        grad = t.gradient(loss, self.G.trainable_variables)
        self.g_opt.apply_gradients(zip(grad, self.G.trainable_variables))
        return loss

    """
    Discriminator training steps using Gradient Tape Graphing
    """

    @tf.function
    def train_d(self, real):
        z = tf.random.normal([2, 1, 32], dtype=tf.dtypes.float32)
        with tf.GradientTape() as t:
            t.watch(z)
            real_logits = self.D(real, training=True)
            x_fake = self.G(z, training=True)
            fake_logits = self.D(x_fake, training=True)
            cost = d_loss_fn(fake_logits, real_logits)
            gp = self.gradient_penalty(partial(self.D, training=True), real, x_fake)
            cost += self.grad_penalty_weight * gp
        grad = t.gradient(cost, self.D.trainable_variables)
        self.d_opt.apply_gradients(zip(grad, self.D.trainable_variables))
        return cost

    """
    Gradient Penalty 
    """

    def gradient_penalty(self, f, real, fake):
        alpha = random.uniform([2, 1, 32], 0., 1.)
        diff = fake - real
        inter = real + (alpha * diff)
        with tf.GradientTape() as t:
            t.watch(inter)
            pred = f(inter)
        grad = t.gradient(pred, [inter])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2]))
        gp = tf.reduce_mean((slopes - 1.) ** 2)
        return gp

    """
    Short Sample Generation using Gaussian Distribution
    """

    @tf.function
    def generate_samples(self):
        z = tf.constant(tf.random.normal([2, 1, 32], dtype=tf.dtypes.float32))
        samples = self.G(z, training=False)
        return samples


"""
Data Pipeline using Tensorflow Dataset Builder
Local caching for efficiency

Parameters:
dataset: Name of selected Dataset contained within Tensorflow Database.
epochs: Number of times dataset will be iterated through.
batch_size: Number of samples selected from the dataset per iteration.
layer_dim: Number of dimensions per layer.
preprocess: Boolean value for preprocessing the dataset.
dataset_info: Information contained within selected dataset.
"""


class DatasetPipeline:
    def __init__(self):
        self.dataset_name = FLAGS.dataset
        self.epochs = FLAGS.epochs
        self.batch_size = FLAGS.batch_size
        self.layer_dim = FLAGS.layer_dim
        self.preprocess = FLAGS.preprocess
        self.dataset_info = []

    def preprocess_label(self, passwords):
        return tf.cast(passwords, tf.int64)

    def dataset_cache(self, dataset):
        tmp_dir = Path(tempfile.gettempdir())
        cache_dir = tmp_dir.joinpath('cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        for p in cache_dir.glob(self.dataset_name + '*'):
            p.unlink()
        return dataset.cache(str(cache_dir / self.dataset_name))

    def load_dataset(self):
        ds, self.dataset_info = tfds.load(name=self.dataset_name,
                                          split='train[:80%]',
                                          with_info=True)

        ds = self.dataset_cache(ds)
        ds = ds.shuffle(50000, reshuffle_each_iteration=True)
        ds = ds.apply(tf.data.experimental.unbatch())
        ds = ds.batch(self.batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds