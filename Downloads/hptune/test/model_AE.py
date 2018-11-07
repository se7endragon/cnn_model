import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from matplotlib.image import cm
from mpl_toolkits.mplot3d import Axes3D

import utils


logging.basicConfig(level=logging.INFO, format='%(message)s')


class AE_mnist():
    def __init__(self, sess, batch_size=100, report_period=100, learning_rate=1e-3,
                 epoch_number=20,          # this is used when we train without using random batch
                 num_iteration=2e+4,  # this is used when we train with using random batch
                 middle_man_dim_1=1000,
                 middle_man_dim_2=1000,
                 latent_space_dim=10,
                 activation_function=tf.nn.relu):
        """
        original_img ---> middle_man_1 ---> middle_man_2 ---> latent_space ---> middle_man_2 ---> middle_man_1 ---> reconstruced_img:
        self.middle_man_dim_1   : dimension of middle_man_1 space
        self.middle_man_dim_2   : dimension of middle_man_2 space
        self.latent_space_dim : dimension of latent_man space
        """
        self.sess = sess
        self.batch_size = int(batch_size)
        self.report_period = int(report_period)
        self.learning_rate = float(learning_rate)
        self.epoch_number = int(epoch_number)
        self.num_iteration = int(num_iteration)
        self.middle_man_dim_1 = int(middle_man_dim_1)
        self.middle_man_dim_2 = int(middle_man_dim_2)
        self.latent_space_dim = int(latent_space_dim)
        self.activation_function = activation_function

    def data_loading(self, data):
        self.data = data

        # unpacking
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_train_cls, self.y_test_cls = self.data

        # data pre-processing
        self.class_names = [0,1,2,3,4,5,6,7,8,9]
        self.num_test = self.y_test.shape[0]

    def encoding(self, x, E_W1, E_b1, E_W2, E_b2, E_W3, E_b3):
        h1 = self.activation_function(tf.matmul(x, E_W1) + E_b1)
        h2 = self.activation_function(tf.matmul(h1, E_W2) + E_b2)
        z = self.activation_function(tf.matmul(h2, E_W3) + E_b3)
        return z

    def decoding(self, z, D_W1, D_b1, D_W2, D_b2, D_W3, D_b3):
        h1 = self.activation_function(tf.matmul(z, D_W1) + D_b1)
        h2 = self.activation_function(tf.matmul(h1, D_W2) + D_b2)
        logits = self.activation_function(tf.matmul(h2, D_W3) + D_b3)
        probs = tf.nn.sigmoid(logits)
        return logits, probs

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
        self.E_W1 = tf.get_variable("E_W1",
                                    shape=(self.img_size_flat, self.middle_man_dim_1),
                                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                    #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.E_b1 = tf.get_variable("E_b1",
                                    shape=(self.middle_man_dim_1, ),
                                    initializer=tf.constant_initializer(0.0))

        self.E_W2 = tf.get_variable("E_W2",
                                    shape=(self.middle_man_dim_1, self.middle_man_dim_2),
                                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                    #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.E_b2 = tf.get_variable("E_b2",
                                    shape=(self.middle_man_dim_2, ),
                                    initializer=tf.constant_initializer(0.0))

        self.E_W3 = tf.get_variable("E_W3",
                                    shape=(self.middle_man_dim_2, self.latent_space_dim),
                                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                    #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.E_b3 = tf.get_variable("E_b3",
                                    shape=(self.latent_space_dim, ),
                                    initializer=tf.constant_initializer(0.0))

        self.D_W1 = tf.get_variable("D_W1",
                                    shape=(self.latent_space_dim, self.middle_man_dim_2),
                                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                    #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.D_b1 = tf.get_variable("D_b1",
                                    shape=(self.middle_man_dim_2, ),
                                    initializer=tf.constant_initializer(0.0))

        self.D_W2 = tf.get_variable("D_W2",
                                    shape=(self.middle_man_dim_2, self.middle_man_dim_1),
                                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                    #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.D_b2 = tf.get_variable("D_b2",
                                    shape=(self.middle_man_dim_1, ),
                                    initializer=tf.constant_initializer(0.0))

        self.D_W3 = tf.get_variable("D_W3",
                                    shape=(self.middle_man_dim_1, self.img_size_flat),
                                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                    #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.D_b3 = tf.get_variable("D_b3",
                                    shape=(self.img_size_flat, ),
                                    initializer=tf.constant_initializer(0.0))

        # encoding and decoding of x
        self.z_x = self.encoding(self.x, self.E_W1, self.E_b1, self.E_W2, self.E_b2, self.E_W3, self.E_b3)
        self.logits_x, self.probs_x = self.decoding(self.z_x, self.D_W1, self.D_b1, self.D_W2, self.D_b2, self.D_W3, self.D_b3)

        # reconstructed images
        self.x_reconstructed = self.probs_x

        # cost and optimizer
        self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_x, labels=self.x)
        self.cost = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # decoding of z
        self.logits_z, self.probs_z  = self.decoding(self.z, self.D_W1, self.D_b1, self.D_W2, self.D_b2, self.D_W3, self.D_b3)

        # generated images
        self.x_generated = self.probs_z

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


class AAE_mnist(AE_mnist):
    def __init__(self, sess, batch_size=100, report_period=1000, learning_rate=1e-3,
                 epoch_number=20,          # this is used when we train without using random batch
                 num_iteration=int(2e+4),  # this is used when we train with using random batch
                 middle_man_dim_1=int(1000),
                 middle_man_dim_2=int(1000),
                 latent_space_dim=int(10),
                 activation_function=tf.nn.relu):
        super().__init__(sess, batch_size, report_period, learning_rate, epoch_number, num_iteration, middle_man_dim_1, middle_man_dim_2, latent_space_dim, activation_function)

    def critic(self, z, C_W1, C_b1, C_W2, C_b2, C_W3, C_b3):
        h1 = self.activation_function(tf.matmul(z, C_W1) + C_b1)
        h2 = self.activation_function(tf.matmul(h1, C_W2) + C_b2)
        logit = tf.matmul(h2, C_W3) + C_b3
        prob = tf.nn.sigmoid(logit)
        return logit, prob

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
        self.E_W1 = tf.get_variable("E_W1",
                                    shape=(self.img_size_flat, self.middle_man_dim_1),
                                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                    #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.E_b1 = tf.get_variable("E_b1",
                                    shape=(self.middle_man_dim_1, ),
                                    initializer=tf.constant_initializer(0.0))

        self.E_W2 = tf.get_variable("E_W2",
                                    shape=(self.middle_man_dim_1, self.middle_man_dim_2),
                                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                    #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.E_b2 = tf.get_variable("E_b2",
                                    shape=(self.middle_man_dim_2, ),
                                    initializer=tf.constant_initializer(0.0))

        self.E_W3 = tf.get_variable("E_W3",
                                    shape=(self.middle_man_dim_2, self.latent_space_dim),
                                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                    #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.E_b3 = tf.get_variable("E_b3",
                                    shape=(self.latent_space_dim, ),
                                    initializer=tf.constant_initializer(0.0))

        self.D_W1 = tf.get_variable("D_W1",
                                    shape=(self.latent_space_dim, self.middle_man_dim_2),
                                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                    #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.D_b1 = tf.get_variable("D_b1",
                                    shape=(self.middle_man_dim_2, ),
                                    initializer=tf.constant_initializer(0.0))

        self.D_W2 = tf.get_variable("D_W2",
                                    shape=(self.middle_man_dim_2, self.middle_man_dim_1),
                                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                    #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.D_b2 = tf.get_variable("D_b2",
                                    shape=(self.middle_man_dim_1, ),
                                    initializer=tf.constant_initializer(0.0))

        self.D_W3 = tf.get_variable("D_W3",
                                    shape=(self.middle_man_dim_1, self.img_size_flat),
                                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                    #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.D_b3 = tf.get_variable("D_b3",
                                    shape=(self.img_size_flat, ),
                                    initializer=tf.constant_initializer(0.0))

        self.C_W1 = tf.get_variable("C_W1",
                                    shape=(self.latent_space_dim, self.middle_man_dim_2),
                                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                    #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.C_b1 = tf.get_variable("C_b1",
                                    shape=(self.middle_man_dim_2, ),
                                    initializer=tf.constant_initializer(0.0))

        self.C_W2 = tf.get_variable("C_W2",
                                    shape=(self.middle_man_dim_2, self.middle_man_dim_1),
                                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                    #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.C_b2 = tf.get_variable("C_b2",
                                    shape=(self.middle_man_dim_1, ),
                                    initializer=tf.constant_initializer(0.0))

        self.C_W3 = tf.get_variable("C_W3",
                                    shape=(self.middle_man_dim_1, 1),
                                    initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                    #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.C_b3 = tf.get_variable("C_b3",
                                    shape=(1, ),
                                    initializer=tf.constant_initializer(0.0))

        # variable grouping
        self.theta_E = [self.E_W1, self.E_b1, self.E_W2, self.E_b2, self.E_W3, self.E_b3]
        self.theta_D = [self.D_W1, self.D_b1, self.D_W2, self.D_b2, self.D_W3, self.D_b3]
        self.theta_C = [self.C_W1, self.C_b1, self.C_W2, self.C_b2, self.C_W3, self.C_b3]

        # encoding, decoding, and critic of x
        self.z_x                    = self.encoding(self.x, self.E_W1, self.E_b1, self.E_W2, self.E_b2, self.E_W3, self.E_b3)
        self.logits_x, self.probs_x = self.decoding(self.z_x, self.D_W1, self.D_b1, self.D_W2, self.D_b2, self.D_W3, self.D_b3)
        self.logit_x,  self.prob_x  = self.critic(self.z_x, self.C_W1, self.C_b1, self.C_W2, self.C_b2, self.C_W3, self.C_b3)

        # reconstructed images
        self.x_reconstructed = self.probs_x

        # decoding and critic of z
        self.logits_z, self.probs_z = self.decoding(self.z, self.D_W1, self.D_b1, self.D_W2, self.D_b2, self.D_W3, self.D_b3)
        self.logit_z,  self.prob_z  = self.critic(self.z, self.C_W1, self.C_b1, self.C_W2, self.C_b2, self.C_W3, self.C_b3)

        # generated images
        self.x_generated = self.probs_z

        # cost and optimizer of AE
        self.cross_entropy_ae = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_x, labels=self.x)
        self.cost_ae = tf.reduce_mean(self.cross_entropy_ae)
        self.optimizer_ae = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost_ae, var_list=self.theta_E+self.theta_D)

        # cost and optimizer of G
        self.cross_entropy_g = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_x, labels=tf.ones_like(self.logit_x))
        self.cost_g = tf.reduce_mean(self.cross_entropy_g)
        self.optimizer_g = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost_g, var_list=self.theta_E)

        # cost and optimizer of C
        self.cross_entropy_c_x = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_x, labels=tf.zeros_like(self.logit_x))
        self.cross_entropy_c_z = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_z, labels=tf.ones_like(self.logit_z))
        self.cost_c = tf.reduce_mean(self.cross_entropy_c_x) + tf.reduce_mean(self.cross_entropy_c_z)
        self.optimizer_c = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost_c, var_list=self.theta_C)

    def train(self):
        for i in range(self.epoch_number):
            start_time = time.time()  # start time of this epoch
            training_batch = zip(range(0, len(self.y_train), self.batch_size),
                                 range(self.batch_size, len(self.y_train), self.batch_size))
            idx_for_print_cost = 0
            for start, end in training_batch:
                feed_dict_ae = {self.x: self.x_train[start:end, :]}
                feed_dict_g  = feed_dict_ae
                feed_dict_c  = {self.x: self.x_train[start:end, :], self.z: np.random.normal(0, 1, [self.batch_size, self.latent_space_dim])}
                self.sess.run(self.optimizer_ae, feed_dict=feed_dict_ae)
                self.sess.run(self.optimizer_g,  feed_dict=feed_dict_g)
                self.sess.run(self.optimizer_c,  feed_dict=feed_dict_c)
                if idx_for_print_cost % self.report_period == 0:                       # for every self.report_period train
                    cost_ae_now = self.sess.run(self.cost_ae, feed_dict=feed_dict_ae)  # we compute cost_ae now
                    cost_g_now  = self.sess.run(self.cost_g,  feed_dict=feed_dict_g)   # we compute cost_g  now
                    cost_c_now  = self.sess.run(self.cost_c,  feed_dict=feed_dict_c)   # we compute cost_c  now
                    print(idx_for_print_cost, cost_ae_now, cost_g_now, cost_d_now)     # we print them
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
            feed_dict_ae = {self.x: x_batch}
            feed_dict_g  = feed_dict_ae
            feed_dict_c  = {self.x: x_batch, self.z: np.random.normal(0, 1, [self.batch_size, self.latent_space_dim])}
            self.sess.run(self.optimizer_ae, feed_dict=feed_dict_ae)
            self.sess.run(self.optimizer_g,  feed_dict=feed_dict_g)
            self.sess.run(self.optimizer_c,  feed_dict=feed_dict_c)
            if (i % self.report_period == 0) or (i == self.num_iteration - 1):
                loss_ae = self.sess.run(self.cost_ae, feed_dict=feed_dict_ae)  # we compute cost_ae now
                loss_g  = self.sess.run(self.cost_g,  feed_dict=feed_dict_g)   # we compute cost_g  now
                loss_c  = self.sess.run(self.cost_c,  feed_dict=feed_dict_c)   # we compute cost_c  now
                logging.info('train iter : {:6d} | loss_ae : {:.6f} | loss_g : {:.6f} | loss_c : {:.6f}'.format(i, loss_ae, loss_g, loss_c))
                self.plot_16_generated_images(figure_save_dir='./img', figure_index=i)


class AE_noisy_bowl(AE_mnist):

    def data_loading(self, data):
        self.data = data

    def graph_construction(self):

        # data dimension
        self.img_size_flat = 3

        # placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_size_flat], name='x')
        self.z = tf.placeholder(tf.float32, shape=[None, self.latent_space_dim], name='z')

        # weights
        self.E_W1 = tf.get_variable("E_W1",
                                    shape=(self.img_size_flat, self.middle_man_dim_1),
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.E_b1 = tf.get_variable("E_b1",
                                    shape=(self.middle_man_dim_1, ),
                                    initializer=tf.constant_initializer(0.0))

        self.E_W2 = tf.get_variable("E_W2",
                                    shape=(self.middle_man_dim_1, self.middle_man_dim_2),
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.E_b2 = tf.get_variable("E_b2",
                                    shape=(self.middle_man_dim_2, ),
                                    initializer=tf.constant_initializer(0.0))

        self.E_W3 = tf.get_variable("E_W3",
                                    shape=(self.middle_man_dim_2, self.latent_space_dim),
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.E_b3 = tf.get_variable("E_b3",
                                    shape=(self.latent_space_dim, ),
                                    initializer=tf.constant_initializer(0.0))

        self.D_W1 = tf.get_variable("D_W1",
                                    shape=(self.latent_space_dim, self.middle_man_dim_2),
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.D_b1 = tf.get_variable("D_b1",
                                    shape=(self.middle_man_dim_2, ),
                                    initializer=tf.constant_initializer(0.0))

        self.D_W2 = tf.get_variable("D_W2",
                                    shape=(self.middle_man_dim_2, self.middle_man_dim_1),
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.D_b2 = tf.get_variable("D_b2",
                                    shape=(self.middle_man_dim_1, ),
                                    initializer=tf.constant_initializer(0.0))

        self.D_W3 = tf.get_variable("D_W3",
                                    shape=(self.middle_man_dim_1, self.img_size_flat),
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.D_b3 = tf.get_variable("D_b3",
                                    shape=(self.img_size_flat, ),
                                    initializer=tf.constant_initializer(0.0))

        # encoding and decoding of x
        self.z_x = self.encoding(self.x, self.E_W1, self.E_b1, self.E_W2, self.E_b2, self.E_W3, self.E_b3)
        self.logits_x, self.probs_x = self.decoding(self.z_x, self.D_W1, self.D_b1, self.D_W2, self.D_b2, self.D_W3, self.D_b3)

        # reconstructed images
        self.x_reconstructed = self.logits_x

        # cost and optimizer
        self.l2_loss = tf.square(self.logits_x - self.x)
        self.cost = tf.reduce_mean(self.l2_loss)
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







