import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import time
from datetime import timedelta
from matplotlib.image import cm
from mpl_toolkits.mplot3d import Axes3D

import utils


logging.basicConfig(level=logging.INFO, format='%(message)s')


class PCA_mnist():
    def __init__(self, sess, batch_size=100, report_period=100, learning_rate=1e-4,
                 epoch_number=100,         # this is used when we train without using random batch
                 num_iteration=2e+4,  # this is used when we train with using random batch
                 latent_space_dim=100):
        """
        original_img --->  latent_space  ---> reconstruced_img:
        self.latent_space_dim : dimension of latent_man space
        """
        self.sess = sess
        self.batch_size = int(batch_size)
        self.report_period = int(report_period)
        self.learning_rate = float(learning_rate)
        self.epoch_number = int(epoch_number)
        self.num_iteration = int(num_iteration)
        self.latent_space_dim = int(latent_space_dim)

    def data_loading(self, data):
        self.data = data

        # data pre-processing
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_train_cls, self.y_test_cls = self.data
        self.class_names = [0,1,2,3,4,5,6,7,8,9]
        self.num_test = self.y_test.shape[0]

    def encoding(self, x, E_W, E_b):
        return tf.matmul(x, E_W) + E_b

    def decoding(self, z, D_W, D_b):
        return tf.matmul(z, D_W) + D_b

    def graph_construction(self):

        # data dimension
        self.img_size = 28
        self.img_size_flat = 784
        self.img_shape = (28, 28)
        self.num_classes = 10

        # placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_size_flat], name='x')
        self.z = tf.placeholder(tf.float32, shape=[None, self.latent_space_dim], name='z')

        # weights
        self.E_W = tf.get_variable("E_W",
                                    shape=(self.img_size_flat, self.latent_space_dim),
                                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                    #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.E_b = tf.get_variable("E_b",
                                    shape=(self.latent_space_dim, ),
                                    initializer=tf.constant_initializer(0.0))

        self.D_W = tf.get_variable("D_W",
                                    shape=(self.latent_space_dim, self.img_size_flat),
                                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                    #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.D_b = tf.get_variable("D_b",
                                    shape=(self.img_size_flat, ),
                                    initializer=tf.constant_initializer(0.0))

        # encoding
        self.x_encoded = self.encoding(self.x, self.E_W, self.E_b)

        # decoding
        self.x_reconstructed = self.decoding(self.x_encoded, self.D_W, self.D_b)

        # cost
        self.cost = tf.nn.l2_loss(self.x - self.x_reconstructed)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # generated image using random sample z in latent space
        self.x_generated  = self.decoding(self.z, self.D_W, self.D_b)

    def train(self):
        for i in range(self.epoch_number):
            start_time = time.time()  # start time of this epoch
            training_batch = zip(range(0, len(self.y_train), self.batch_size),
                                 range(self.batch_size, len(self.y_train), self.batch_size))
            idx_for_print_cost = 0
            for start, end in training_batch:
                feed_dict = {self.x: self.x_train[start:end, :]}
                self.sess.run(self.optimizer, feed_dict=feed_dict)
                if idx_for_print_cost % self.report_period == 0:              # for every self.report_period train
                    cost_now = self.sess.run(self.cost, feed_dict=feed_dict)  # we compute cost now
                    print(idx_for_print_cost, cost_now)                       # we print cost now
                idx_for_print_cost += 1
            end_time = time.time()  # end time of this epoch

            print("==========================================================")
            print("Epoch:", i)
            time_dif = end_time - start_time                                     # we check computing time for each epoch
            print("Time Usage: " + str(timedelta(seconds=int(round(time_dif))))) # and print it
            self.plot_16_generated_images(figure_save_dir='./img', figure_index=i)

    def train_random_batch(self):
        for i in range(self.num_iteration):
            idx = np.random.choice(self.x_train.shape[0], size=self.batch_size, replace=False)  # random_batch
            x_batch = self.x_train[idx]                                                         # random_batch
            feed_dict = {self.x: x_batch}
            self.sess.run(self.optimizer, feed_dict=feed_dict)
            if (i % self.report_period == 0) or (i == self.num_iteration - 1):
                loss = self.sess.run(self.cost, feed_dict=feed_dict)
                logging.info('train iter : {:6d} | loss : {:.6f}'.format(i, loss))
                self.plot_16_generated_images(figure_save_dir='./img', figure_index=i)

    def plot_16_generated_images(self, figure_save_dir, figure_index):
        if not os.path.exists(figure_save_dir):
            os.makedirs(figure_save_dir)

        feed_dict = {self.z: np.random.normal(0, 1, size=(16, self.latent_space_dim))}
        images = self.sess.run(self.x_generated, feed_dict=feed_dict)

        fig = utils.plot_16_images_2d_and_returen(images, img_shape=self.img_shape)
        plt.savefig(figure_save_dir + '/{}.png'.format(figure_index), bbox_inches='tight')
        plt.close(fig)

    def visualization_of_reconstruction(self):
        imgs_original = self.x_test[0:16, :]
        feed_dict = {self.x: imgs_original}
        imgs_recon = self.sess.run(self.x_reconstructed, feed_dict=feed_dict)

        fig = utils.plot_16_images_2d_and_returen(imgs_original, img_shape=self.img_shape)
        plt.show(fig)
        fig = utils.plot_16_images_2d_and_returen(imgs_recon, img_shape=self.img_shape)
        plt.show(fig)

    def visualization_of_16_loading_vectors(self):
        z_batch = np.zeros(shape=(16, self.latent_space_dim))
        for i in range(16):
            if i < self.latent_space_dim:
                z_batch[i, i] = 1

        feed_dict = {self.z: z_batch}
        images = self.sess.run(self.x_generated, feed_dict=feed_dict)

        fig = utils.plot_16_images_2d_and_returen(images, img_shape=self.img_shape)
        plt.show(fig)

    def visualization_of_zero_vector_in_latent_space(self):
        z_batch = np.zeros(shape=(1, self.latent_space_dim))
        feed_dict = {self.z: z_batch}
        img = self.sess.run(self.x_generated, feed_dict=feed_dict)

        fig = utils.plot_one_image(img, self.img_shape)
        plt.show(fig)

    def save(self, sess, save_path):
        self.saver = tf.train.Saver()
        self.sess = sess
        self.save_path = save_path
        self.save_dir = self.save_path.split('/')[0]

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        self.saver.save(sess=self.sess, save_path=self.save_path)
        print("Graph Saved")

    def restore(self, sess, save_path):
        self.saver = tf.train.Saver()
        self.sess = sess
        self.save_path = save_path
        self.save_dir = self.save_path.split('/')[0]

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        self.saver.restore(sess=self.sess, save_path=self.save_path)
        print("Graph Restored")


class PCA_noisy_bowl(PCA_mnist):

    def data_loading(self, data):
        self.data = data

    def encoding(self, x, E_W, E_b):
        return tf.matmul(x, E_W) + E_b

    def decoding(self, z, D_W, D_b):
        return tf.matmul(z, D_W) + D_b

    def graph_construction(self):

        # data dimension
        self.img_size_flat = 3

        # placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_size_flat], name='x')
        self.z = tf.placeholder(tf.float32, shape=[None, self.latent_space_dim], name='z')

        # weights
        self.E_W = tf.get_variable("E_W",
                                    shape=(self.img_size_flat, self.latent_space_dim),
                                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                    #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.E_b = tf.get_variable("E_b",
                                    shape=(self.latent_space_dim, ),
                                    initializer=tf.constant_initializer(0.0))

        self.D_W = tf.get_variable("D_W",
                                    shape=(self.latent_space_dim, self.img_size_flat),
                                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                    #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.D_b = tf.get_variable("D_b",
                                    shape=(self.img_size_flat, ),
                                    initializer=tf.constant_initializer(0.0))

        # encoding
        self.x_encoded = self.encoding(self.x, self.E_W, self.E_b)

        # decoding
        self.x_reconstructed = self.decoding(self.x_encoded, self.D_W, self.D_b)

        # cost
        self.cost = tf.nn.l2_loss(self.x - self.x_reconstructed)

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def train(self):
        for i in range(self.epoch_number):
            start_time = time.time()  # start time of this epoch
            training_batch = zip(range(0, len(self.data), self.batch_size),
                                 range(self.batch_size, len(self.data), self.batch_size))
            idx_for_print_cost = 0
            for start, end in training_batch:
                feed_dict = {self.x: self.data[start:end, :]}
                self.sess.run(self.optimizer, feed_dict=feed_dict)
                if idx_for_print_cost % self.report_period == 0:              # for every self.report_period train
                    cost_now = self.sess.run(self.cost, feed_dict=feed_dict)  # we compute cost now
                    print(idx_for_print_cost, cost_now)                       # we print cost now
                idx_for_print_cost += 1
            end_time = time.time()  # end time of this epoch

            print("==========================================================")
            print("Epoch:", i)
            time_dif = end_time - start_time                                     # we check computing time for each epoch
            print("Time Usage: " + str(timedelta(seconds=int(round(time_dif))))) # and print it

    def train_random_batch(self):
        for i in range(self.num_iteration):
            idx = np.random.choice(len(self.data), size=self.batch_size, replace=False)  # random_batch
            x_batch = self.data[idx]                                                         # random_batch
            feed_dict = {self.x: x_batch}
            self.sess.run(self.optimizer, feed_dict=feed_dict)
            if (i % self.report_period == 0) or (i == self.num_iteration - 1):
                loss = self.sess.run(self.cost, feed_dict=feed_dict)
                logging.info('train iter : {:6d} | loss : {:.6f}'.format(i, loss))

    def visualization_of_reconstruction(self):
        imgs_original = self.data
        feed_dict = {self.x: imgs_original}
        imgs_recon = self.sess.run(self.x_reconstructed, feed_dict=feed_dict)

        x_grid = imgs_recon[:, 0].reshape(40, 40)
        y_grid = imgs_recon[:, 1].reshape(40, 40)
        z_grid = imgs_recon[:, 2].reshape(40, 40)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(x_grid, y_grid, z_grid, rstride=1, cstride=1,
                    cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_title('Smooth Bowl')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.0, 2.5])
        plt.show()
