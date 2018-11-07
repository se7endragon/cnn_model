import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

import utils


logging.basicConfig(level=logging.INFO, format='%(message)s')


class VAE_mnist():
    def __init__(self, sess, batch_size=100, report_period=100, learning_rate=1e-3,
                 epoch_number=20,          # this is used when we train without using random batch
                 num_iteration=2e+4,  # this is used when we train with using random batch
                 middle_man_dim_1=1000,
                 middle_man_dim_2=1000,
                 latent_space_dim=10,
                 activation_function=tf.nn.relu,
                 alpha=0.5):
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
        self.alpha = float(alpha)

    def data_loading(self, data):
        self.data = data

        # data pre-processing
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_train_cls, self.y_test_cls = self.data
        self.class_names = [0,1,2,3,4,5,6,7,8,9]
        self.num_test = self.y_test.shape[0]

    def encoding(self, x, E_W1, E_b1, E_W2, E_b2, E_W3_mu, E_b3_mu, E_W3_log_var, E_b3_log_var):
        h1 = self.activation_function(tf.matmul(x, E_W1) + E_b1)
        h2 = self.activation_function(tf.matmul(h1, E_W2) + E_b2)
        mu = tf.matmul(h2, E_W3_mu) + E_b3_mu
        log_var = tf.matmul(h2, E_W3_log_var) + E_b3_log_var
        return mu, log_var

    def sampling(self, mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        z = mu + tf.exp(log_var / 2) * eps
        return z

    def decoding(self, z, D_W1, D_b1, D_W2, D_b2, D_W3, D_b3):
        h1 = self.activation_function(tf.matmul(z, D_W1) + D_b1)
        h2 = self.activation_function(tf.matmul(h1, D_W2) + D_b2)
        logits = tf.matmul(h2, D_W3) + D_b3
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

        self.E_W3_mu = tf.get_variable("E_W3_mu",
                                       shape=(self.middle_man_dim_2, self.latent_space_dim),
                                       initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                       #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.E_b3_mu = tf.get_variable("E_b3_mu",
                                       shape=(self.latent_space_dim, ),
                                       initializer=tf.constant_initializer(0.0))

        self.E_W3_log_var = tf.get_variable("E_W3_log_var",
                                            shape=(self.middle_man_dim_2, self.latent_space_dim),
                                            initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                            #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.E_b3_log_var = tf.get_variable("E_b3_log_var",
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

        # encoding, sampling, and decoding of x
        self.mu_x, self.log_var_x = self.encoding(self.x, self.E_W1, self.E_b1, self.E_W2, self.E_b2, self.E_W3_mu, self.E_b3_mu, self.E_W3_log_var, self.E_b3_log_var)
        self.z_x = self.sampling(self.mu_x, self.log_var_x)
        self.logits_x, self.probs_x = self.decoding(self.z_x, self.D_W1, self.D_b1, self.D_W2, self.D_b2, self.D_W3, self.D_b3)

        # reconstructed images
        self.x_reconstructed = self.probs_x

        # cost and optimizer
        self.cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_x, labels=self.x), axis=1)
        self.cost_ce = tf.reduce_mean(self.cross_entropy)
        self.kl_divergence = tf.reduce_sum(tf.exp(self.log_var_x) + self.mu_x**2 - 1. - self.log_var_x, axis=1)
        self.cost_kl = tf.reduce_mean(self.kl_divergence)
        self.cost = self.cost_ce + self.alpha * self.cost_kl
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # decoding of z
        self.logits_z, self.probs_z = self.decoding(self.z, self.D_W1, self.D_b1, self.D_W2, self.D_b2, self.D_W3, self.D_b3)

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


class DVAE_mnist(VAE_mnist):
    def __init__(self, sess, batch_size=100, report_period=100, learning_rate=1e-3,
                 epoch_number=20,          # this is used when we train without using random batch
                 num_iteration=int(2e+4),  # this is used when we train with using random batch
                 middle_man_dim_1=int(1000),
                 middle_man_dim_2=int(1000),
                 latent_space_dim=int(10),
                 activation_function=tf.nn.relu,
                 alpha=0.5,
                 noise_factor=0.1):
        super().__init__(sess, batch_size, report_period, learning_rate, epoch_number, num_iteration, middle_man_dim_1,
                         middle_man_dim_2, latent_space_dim, activation_function, alpha)
        self.noise_factor = float(noise_factor)

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

        self.E_W3_mu = tf.get_variable("E_W3_mu",
                                       shape=(self.middle_man_dim_2, self.latent_space_dim),
                                       initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                       #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.E_b3_mu = tf.get_variable("E_b3_mu",
                                       shape=(self.latent_space_dim, ),
                                       initializer=tf.constant_initializer(0.0))

        self.E_W3_log_var = tf.get_variable("E_W3_log_var",
                                            shape=(self.middle_man_dim_2, self.latent_space_dim),
                                            initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                            #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.E_b3_log_var = tf.get_variable("E_b3_log_var",
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

        # encoding, sampling, and decoding of x
        # Add noise to X
        self.x_noise = self.x + self.noise_factor * tf.random_normal(tf.shape(self.x))
        self.x_noise_clipped = tf.clip_by_value(self.x_noise, 0., 1.)

        self.mu_x, self.log_var_x = self.encoding(self.x_noise_clipped, self.E_W1, self.E_b1, self.E_W2, self.E_b2, self.E_W3_mu, self.E_b3_mu, self.E_W3_log_var, self.E_b3_log_var)
        self.z_x = self.sampling(self.mu_x, self.log_var_x)
        self.logits_x, self.probs_x = self.decoding(self.z_x, self.D_W1, self.D_b1, self.D_W2, self.D_b2, self.D_W3, self.D_b3)

        # reconstructed images
        self.x_reconstructed = self.probs_x

        # cost and optimizer
        self.cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_x, labels=self.x), axis=1)
        self.cost_ce = tf.reduce_mean(self.cross_entropy)
        self.kl_divergence = tf.reduce_sum(tf.exp(self.log_var_x) + self.mu_x**2 - 1. - self.log_var_x, axis=1)
        self.cost_kl = tf.reduce_mean(self.kl_divergence)
        self.cost = self.cost_ce + self.alpha * self.cost_kl
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # decoding of z
        self.logits_z, self.probs_z = self.decoding(self.z, self.D_W1, self.D_b1, self.D_W2, self.D_b2, self.D_W3, self.D_b3)

        # generated images
        self.x_generated = self.probs_z


class CVAE_mnist(VAE_mnist):
    def __init__(self, sess, batch_size=100, report_period=100, learning_rate=1e-3,
                 epoch_number=20,          # this is used when we train without using random batch
                 num_iteration=int(2e+4),  # this is used when we train with using random batch
                 middle_man_dim_1=int(1000),
                 middle_man_dim_2=int(1000),
                 latent_space_dim=int(10),
                 activation_function=tf.nn.relu,
                 alpha=0.5):
        super().__init__(sess, batch_size, report_period, learning_rate, epoch_number, num_iteration, middle_man_dim_1,
                         middle_man_dim_2, latent_space_dim, activation_function, alpha)

    def encoding(self, x, y, E_W1, E_b1, E_W2, E_b2, E_W3_mu, E_b3_mu, E_W3_log_var, E_b3_log_var):
        inputs = tf.concat(axis=1, values=[x, y])
        h1 = self.activation_function(tf.matmul(inputs, E_W1) + E_b1)
        h2 = self.activation_function(tf.matmul(h1, E_W2) + E_b2)
        mu = tf.matmul(h2, E_W3_mu) + E_b3_mu
        log_var = tf.matmul(h2, E_W3_log_var) + E_b3_log_var
        return mu, log_var

    def sampling(self, mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        z = mu + tf.exp(log_var / 2) * eps
        return z

    def decoding(self, z, y, D_W1, D_b1, D_W2, D_b2, D_W3, D_b3):
        inputs = tf.concat(axis=1, values=[z, y])
        h1 = self.activation_function(tf.matmul(inputs, D_W1) + D_b1)
        h2 = self.activation_function(tf.matmul(h1, D_W2) + D_b2)
        logits = tf.matmul(h2, D_W3) + D_b3
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
        self.y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y')
        self.z = tf.placeholder(tf.float32, shape=[None, self.latent_space_dim], name='z')

        # weights
        self.E_W1 = tf.get_variable("E_W1",
                                    shape=(self.img_size_flat + self.num_classes, self.middle_man_dim_1),
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

        self.E_W3_mu = tf.get_variable("E_W3_mu",
                                       shape=(self.middle_man_dim_2, self.latent_space_dim),
                                       initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                       #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.E_b3_mu = tf.get_variable("E_b3_mu",
                                       shape=(self.latent_space_dim, ),
                                       initializer=tf.constant_initializer(0.0))

        self.E_W3_log_var = tf.get_variable("E_W3_log_var",
                                            shape=(self.middle_man_dim_2, self.latent_space_dim),
                                            initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                            #initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.E_b3_log_var = tf.get_variable("E_b3_log_var",
                                            shape=(self.latent_space_dim, ),
                                            initializer=tf.constant_initializer(0.0))

        self.D_W1 = tf.get_variable("D_W1",
                                    shape=(self.latent_space_dim + self.num_classes, self.middle_man_dim_2),
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

        # encoding, sampling, and decoding of x
        self.mu_x, self.log_var_x = self.encoding(self.x, self.y, self.E_W1, self.E_b1, self.E_W2, self.E_b2, self.E_W3_mu, self.E_b3_mu, self.E_W3_log_var, self.E_b3_log_var)
        self.z_x = self.sampling(self.mu_x, self.log_var_x)
        self.logits_x, self.probs_x = self.decoding(self.z_x, self.y, self.D_W1, self.D_b1, self.D_W2, self.D_b2, self.D_W3, self.D_b3)

        # reconstructed images
        self.x_reconstructed = self.probs_x

        # cost and optimizer
        self.cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_x, labels=self.x), axis=1)
        self.cost_ce = tf.reduce_mean(self.cross_entropy)
        self.kl_divergence = tf.reduce_sum(tf.exp(self.log_var_x) + self.mu_x**2 - 1. - self.log_var_x, axis=1)
        self.cost_kl = tf.reduce_mean(self.kl_divergence)
        self.cost = self.cost_ce + self.alpha * self.cost_kl
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # decoding of z
        self.logits_z, self.probs_z = self.decoding(self.z, self.y, self.D_W1, self.D_b1, self.D_W2, self.D_b2, self.D_W3, self.D_b3)

        # generated images
        self.x_generated = self.probs_z

    def train(self):
        for i in range(self.epoch_number):
            start_time = time.time()  # start time of this epoch
            training_batch = zip(range(0, len(self.y_train), self.batch_size),
                                 range(self.batch_size, len(self.y_train), self.batch_size))
            idx_for_print_cost = 0
            for start, end in training_batch:
                feed_dict = {self.x: self.x_train[start:end, :],
                             self.y: self.y_train[start:end, :]}
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
            x_batch = self.x_train[idx]
            y_batch = self.y_train[idx]                                                        # random_batch
            feed_dict = {self.x: x_batch,
                         self.y: y_batch}
            self.sess.run(self.optimizer, feed_dict=feed_dict)
            if (i % self.report_period == 0) or (i == self.num_iteration - 1):
                loss = self.sess.run(self.cost, feed_dict=feed_dict)
                logging.info('train iter : {:6d} | loss : {:.6f}'.format(i, loss))

    def visualization_of_reconstruction(self):
        imgs_original = self.x_test[0:16, :]
        labels_original = self.y_test[0:16, :]
        feed_dict = {self.x: imgs_original,
                     self.y: labels_original}
        imgs_recon = self.sess.run(self.x_reconstructed, feed_dict=feed_dict)

        fig = utils.plot_16_images_2d_and_returen(imgs_original, img_shape=self.img_shape)
        plt.show(fig)
        fig = utils.plot_16_images_2d_and_returen(imgs_recon, img_shape=self.img_shape)
        plt.show(fig)

    def visualization_of_16_loading_vectors(self):
        z_batch = np.zeros(shape=(16, self.latent_space_dim))
        y_batch = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]).astype(np.float32)
        for i in range(16):
            if i < self.latent_space_dim:
                z_batch[i, i] = 1

        feed_dict = {self.z: z_batch,
                     self.y: y_batch}
        images = self.sess.run(self.x_generated, feed_dict=feed_dict)

        fig = utils.plot_16_images_2d_and_returen(images, img_shape=self.img_shape)
        plt.show(fig)

    def visualization_of_zero_vector_in_latent_space(self):
        z_batch = np.zeros(shape=(1, self.latent_space_dim))
        y_batch = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).astype(np.float32)
        feed_dict = {self.z: z_batch,
                     self.y: y_batch}
        img = self.sess.run(self.x_generated, feed_dict=feed_dict)

        fig = utils.plot_one_image(img, self.img_shape)
        plt.show(fig)







