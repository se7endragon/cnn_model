import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import logging
import os
import time
from datetime import timedelta

import utils


logging.basicConfig(level=logging.INFO, format='%(message)s')


class CNN_mnist():
    def __init__(self, sess, batch_size=128, report_period=100, learning_rate=1e-3,
                 epoch_number=2,           # this is used when we train without using random batch
                 num_iteration=int(1e+3)): # this is used when we train with using random batch
        self.sess = sess
        self.batch_size = int(batch_size)
        self.report_period = int(report_period)
        self.learning_rate = float(learning_rate)
        self.epoch_number = int(epoch_number)
        self.num_iteration = int(num_iteration)

    def data_loading(self, data):
        self.data = data

        # data pre-processing
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_train_cls, self.y_test_cls = self.data
        self.class_names = [0,1,2,3,4,5,6,7,8,9]
        self.num_test = self.y_test.shape[0]

    def convolution(self, input_tensor, conv_W, conv_b, batch_norm=False):
        layer = tf.nn.conv2d(input=input_tensor,
                             filter=conv_W,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        layer += conv_b
        layer_pooled = tf.nn.max_pool(value=layer,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME')
        layer_pooled_relu = tf.nn.relu(layer_pooled)
        if batch_norm:
            output_tensor = tf.contrib.layers.batch_norm(layer_pooled_relu,
                                                         center=True, scale=True, is_training=self.is_train)
        else:
            output_tensor = layer_pooled_relu

        return output_tensor

    def fully_connected(self, flat_input_tensor, fc_W, fc_b, relu=True, dropout=True):
        layer = tf.matmul(flat_input_tensor, fc_W) + fc_b
        if relu:
            layer = tf.nn.relu(layer)
        if dropout:
            layer = tf.nn.dropout(layer, keep_prob=self.keep_prob)
        return layer

    def graph_construction(self):

        # graph dimension
        self.num_filters1 = 32   # Convolutional Layer 1: There are 32 of these filters.
        self.num_filters2 = 64   # Convolutional Layer 2: There are 64 of these filters.
        self.fc_size      = 1024 # Fully-connected layer: Number of neurons in fully-connected layer.
        self.filter_size1 = 5    # Convolutional Layer 1: Convolution filters are 5 x 5 pixels.
        self.filter_size2 = 5    # Convolutional Layer 2: Convolution filters are 5 x 5 pixels.
        self.num_features = 3136 # number of features in self.layer2_flat

        # data dimension
        self.img_size = 28
        self.img_shape = (28, 28)
        self.num_classes = 10
        self.num_channels = 1

        # placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.num_channels], name='x')
        self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')
        self.y_true_cls = tf.placeholder(tf.int32, shape=[None], name='y_true_cls')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.is_train = tf.placeholder(tf.bool, shape=[], name='is_train')

        # weights
        self.conv1_W = tf.get_variable("conv1_W",
                                       shape=(self.filter_size1, self.filter_size1, self.num_channels, self.num_filters1),
                                       #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.conv1_b = tf.get_variable("conv1_b",
                                       shape=(self.num_filters1, ),
                                       initializer=tf.constant_initializer(0.0))

        self.conv2_W = tf.get_variable("conv2_W",
                                       shape=(self.filter_size2, self.filter_size2, self.num_filters1, self.num_filters2),
                                       #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.conv2_b = tf.get_variable("conv2_b",
                                       shape=(self.num_filters2, ),
                                       initializer=tf.constant_initializer(0.0))

        self.fc1_W = tf.get_variable("fc1_W",
                                     shape=(self.num_features, self.fc_size),
                                     #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.fc1_b = tf.get_variable("fc1_b",
                                     shape=(self.fc_size, ),
                                     initializer=tf.constant_initializer(0.0))

        self.fc2_W = tf.get_variable("fc2_W",
                                     shape=(self.fc_size, self.num_classes),
                                     #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.fc2_b = tf.get_variable("fc2_b",
                                     shape=(self.num_classes, ),
                                     initializer=tf.constant_initializer(0.0))

        # convolution layers
        self.layer1_out = self.convolution(self.x, self.conv1_W, self.conv1_b)           # we don't use batch_norm
        self.layer2_out = self.convolution(self.layer1_out, self.conv2_W, self.conv2_b)  # we don't use batch_norm

        # flatten output of 2nd convolution layer
        self.layer2_out_shape = self.layer2_out.get_shape()                     # self.num_features = 3136 given above explicitly
        self.num_features = self.layer2_out_shape[1:4].num_elements()           # we need this number explicitly to define self.fc1_W above
        self.layer2_flat = tf.reshape(self.layer2_out, [-1, self.num_features]) # otherwise we have to define self.fc1_W right here

        # fully connected layers
        self.fc1_out = self.fully_connected(self.layer2_flat, self.fc1_W, self.fc1_b, relu=True, dropout=True)

        # logits and related
        self.logits = self.fully_connected(self.fc1_out, self.fc2_W, self.fc2_b, relu=False, dropout=False)
        self.y_pred = tf.nn.softmax(self.logits, name='y_pred')
        self.y_pred_cls = tf.cast(tf.argmax(self.logits, axis=1), tf.int32)

        # cost and optimizer
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                                                                        labels=self.y_true)
        self.cost = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # performance measure
        self.correct_bool = tf.equal(self.y_pred_cls, self.y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_bool, tf.float32))

    def train(self, is_print=True):
        for i in range(self.epoch_number):
            start_time = time.time() # start time of this epoch
            training_batch = zip(range(0, len(self.y_train), self.batch_size),
                                 range(self.batch_size, len(self.y_train), self.batch_size))
            idx_for_print_cost = 0
            for start, end in training_batch:                         # for each train
                feed_dict = {self.x: self.x_train[start:end,:],       # we pick up train batch sequentially
                             self.y_true: self.y_train[start:end,:],  # we don't use random batch
                             self.keep_prob: 0.5,                     # we use dropout with self.keep_prob: 0.5
                             self.is_train: True}                     # during train period
                self.sess.run(self.optimizer, feed_dict=feed_dict)    # here is the main excution of train
                if (idx_for_print_cost % self.report_period == 0) and is_print:  # for every self.report_period train
                    cost_now = self.sess.run(self.cost, feed_dict=feed_dict)  # we compute cost now
                    print(idx_for_print_cost, cost_now)                       # we print cost now
                idx_for_print_cost += 1
            end_time = time.time() # end time of this epoch

            if is_print:
                print("==========================================================")
                print("Epoch:", i)
                test_indices = np.arange(len(self.y_test))                                                       # at the end of each epoch
                np.random.shuffle(test_indices)                                                                  # we choose random batch from test set
                test_indices = test_indices[0:self.batch_size]                                                   # and compute accuracy on this random test set
                feed_dict = {self.x: self.x_test[:self.batch_size,:],                                            #
                             self.y_true: self.y_test[:self.batch_size,:],                                       #
                             self.y_true_cls: self.y_test_cls[:self.batch_size],                                 #
                             self.keep_prob: 1.0,                                                                #
                             self.is_train: False}                                                               #
                print("Accuracy on Random Test Samples: %g" % self.sess.run(self.accuracy, feed_dict=feed_dict)) #
                time_dif = end_time - start_time                                     # we check computing time for each epoch
                print("Time Usage: " + str(timedelta(seconds=int(round(time_dif))))) # and print it

    def train_random_batch(self):
        for i in range(self.num_iteration):
            idx = np.random.choice(self.x_train.shape[0], size=self.batch_size, replace=False) # random_batch
            x_batch, y_batch = self.x_train[idx], self.y_train[idx]                            # random_batch
            feed_dict = {self.x: x_batch,
                         self.y_true: y_batch,
                         self.keep_prob: 0.5,
                         self.is_train: True}
            self.sess.run(self.optimizer, feed_dict=feed_dict)
            if (i % self.report_period == 0) or (i == self.num_iteration - 1):
                loss = self.sess.run(self.cost, feed_dict=feed_dict)
                logging.info('train iter : {:6d} | loss : {:.6f}'.format(i, loss))

    def print_test_accuracy(self, is_print=True):
        cls_pred = np.zeros(shape=self.num_test, dtype=np.int32)

        i = 0
        while i < self.num_test:
            j = min(i + self.batch_size, self.num_test)
            self.test_images = self.x_test[i:j, :]
            feed_dict = {self.x: self.test_images, self.keep_prob: 1.0, self.is_train: False}
            cls_pred[i:j] = self.sess.run(self.y_pred_cls, feed_dict=feed_dict)
            i = j

        test_accuracy = (cls_pred == self.y_test_cls).sum() / self.num_test

        if is_print:
            print("Accuracy on test-set: {0:.1%}".format(test_accuracy))

        return test_accuracy

    def get_test_probs(self):
        test_probs = np.zeros(shape=[self.num_test, self.num_classes], dtype=np.float32)

        i = 0
        while i < self.num_test:
            j = min(i + self.batch_size, self.num_test)
            self.test_images = self.x_test[i:j, :]
            feed_dict = {self.x: self.test_images, self.keep_prob: 1.0, self.is_train: False}
            test_probs[i:j] = self.sess.run(self.y_pred, feed_dict=feed_dict)
            i = j

        return test_probs

    def print_confusion_matrix(self):
        cls_pred = np.zeros(shape=self.num_test, dtype=np.int)

        i = 0
        while i < self.num_test:
            j = min(i + self.batch_size, self.num_test)
            self.test_images = self.x_test[i:j, :]
            self.test_labels = self.y_test[i:j, :]
            feed_dict = {self.x: self.test_images,
                         self.y_true: self.test_labels,
                         self.keep_prob: 1.0,
                         self.is_train: False}
            cls_pred[i:j] = self.sess.run(self.y_pred_cls, feed_dict=feed_dict)
            i = j

        cm = confusion_matrix(y_true=self.y_test_cls,
                              y_pred=cls_pred)
        print(cm)

        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.colorbar()
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, range(self.num_classes))
        plt.yticks(tick_marks, range(self.num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        plt.savefig('img/cm.png')
        plt.close()

    def plot_9_test_images_with_false_prediction(self):
        images_false_prediction = []
        cls_true = []
        cls_pred = []
        num_false_prediction = 0
        i = 0
        while num_false_prediction < 9:
            feed_dict = {self.x: [self.x_test[i]],
                         self.y_true: [self.y_test[i]],
                         self.y_true_cls: [self.y_test_cls[i]],
                         self.keep_prob: 1.0,
                         self.is_train: False}
            y_pred_cls, correct_bool = self.sess.run([self.y_pred_cls, self.correct_bool],
                                                     feed_dict=feed_dict)
            if correct_bool == False:
                images_false_prediction.append(self.x_test[i])
                cls_true.append(self.class_names[self.y_test_cls[i]])
                cls_pred.append(self.class_names[y_pred_cls[0]])

                num_false_prediction += 1
            i += 1

        utils.plot_many_images_2d(images=images_false_prediction,
                                  img_shape=self.img_shape,
                                  cls_true=cls_true,
                                  cls_pred=cls_pred)

    def save(self, sess, save_path, is_print=True):
        self.saver = tf.train.Saver()
        self.sess = sess
        self.save_path = save_path
        self.save_dir = self.save_path.split('/')[0]

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        self.saver.save(sess=self.sess, save_path=self.save_path)

        if is_print:
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

    def plot_test_images_in_input_conv1_conv2_layers(self, num_test_images=2):
        for i in range(num_test_images):
            image0 = self.x_test[i]
            utils.plot_one_image(image0, self.img_shape, cls_true=None, cls_pred=None)

            feed_dict = {self.x: [image0],
                         self.keep_prob: 1.0,
                         self.is_train: False}
            image0_layer1, image0_layer2 = self.sess.run([self.layer1_out, self.layer2_out],
                                                          feed_dict=feed_dict)

            image0_layer1 = image0_layer1[0, :, :, :]
            image0_layer2 = image0_layer2[0, :, :, :]

            utils.plot_many_images_3d(image0_layer1)
            utils.plot_many_images_3d(image0_layer2)


class CNN_mnist_adversary(CNN_mnist):
    def __init__(self, sess, batch_size=128, report_period=100,
                 learning_rate=1e-3,
                 learning_rate_adversary=1e-2,
                 epoch_number=2,            # this is used when we train without using random batch
                 epoch_number_adversary=2,  # this is used when we train without using random batch
                 num_iteration=1e+3,           # this is used when we train with using random batch
                 num_iteration_adversary=5e+3, # this is used when we train with using random batch
                 noise_limit=0.35,    # noise is between -noise_limit and noise_limit
                 noise_weight=0.02):  # cost_adversary = cost + noise_weight * tf.nn.l2_loss(nnoise)
        super().__init__(sess, batch_size, report_period, learning_rate, epoch_number, num_iteration)
        self.learning_rate_adversary = float(learning_rate_adversary)
        self.epoch_number_adversary = int(epoch_number_adversary)
        self.num_iteration_adversary = int(num_iteration_adversary)
        self.noise_limit = float(noise_limit)
        self.noise_weight = float(noise_weight)

    def graph_construction(self):

        # graph dimension
        self.num_filters1 = 32   # Convolutional Layer 1: There are 32 of these filters.
        self.num_filters2 = 64   # Convolutional Layer 2: There are 64 of these filters.
        self.fc_size      = 1024 # Fully-connected layer: Number of neurons in fully-connected layer.
        self.filter_size1 = 5    # Convolutional Layer 1: Convolution filters are 5 x 5 pixels.
        self.filter_size2 = 5    # Convolutional Layer 2: Convolution filters are 5 x 5 pixels.
        self.num_features = 3136 # number of features in self.layer2_flat

        # data dimension
        self.img_size = 28
        self.img_shape = (28, 28)
        self.num_classes = 10
        self.num_channels = 1

        # placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.num_channels], name='x')
        self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')
        self.y_true_cls = tf.placeholder(tf.int32, shape=[None], name='y_true_cls')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.is_train = tf.placeholder(tf.bool, shape=[], name='is_train')

        # add noise
        self.noise = tf.get_variable('noise',
                                     shape=(self.img_size, self.img_size, self.num_channels),
                                     initializer=tf.constant_initializer(0.0))
        self.x_noisy_image = tf.clip_by_value(self.x + self.noise, 0.0, 1.0)
        # The adversarial noise will be limited / clipped to the given ± noise-limit that we set above.
        # Note that this is actually not executed at this point in the computational graph,
        # but will instead be executed after the optimization-step, see further below.
        self.noise_clip = tf.assign(self.noise, tf.clip_by_value(self.noise,
                                                                 -self.noise_limit,
                                                                 self.noise_limit))

        self.var_list_adversary = [self.noise]

        # weights
        self.conv1_W = tf.get_variable("conv1_W",
                                       shape=(self.filter_size1, self.filter_size1, self.num_channels, self.num_filters1),
                                       #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.conv1_b = tf.get_variable("conv1_b",
                                       shape=(self.num_filters1, ),
                                       initializer=tf.constant_initializer(0.0))

        self.conv2_W = tf.get_variable("conv2_W",
                                       shape=(self.filter_size2, self.filter_size2, self.num_filters1, self.num_filters2),
                                       #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.conv2_b = tf.get_variable("conv2_b",
                                       shape=(self.num_filters2, ),
                                       initializer=tf.constant_initializer(0.0))

        self.fc1_W = tf.get_variable("fc1_W",
                                     shape=(self.num_features, self.fc_size),
                                     #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.fc1_b = tf.get_variable("fc1_b",
                                     shape=(self.fc_size, ),
                                     initializer=tf.constant_initializer(0.0))

        self.fc2_W = tf.get_variable("fc2_W",
                                     shape=(self.fc_size, self.num_classes),
                                     #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.fc2_b = tf.get_variable("fc2_b",
                                     shape=(self.num_classes, ),
                                     initializer=tf.constant_initializer(0.0))

        self.var_list = [self.conv1_W, self.conv1_b,
                         self.conv2_W, self.conv2_b,
                         self.fc1_W, self.fc1_b,
                         self.fc2_W, self.fc2_b]

        # convolution layers
        self.layer1_out = self.convolution(self.x_noisy_image, self.conv1_W, self.conv1_b) # we don't use batch_norm
        self.layer2_out = self.convolution(self.layer1_out, self.conv2_W, self.conv2_b)    # we don't use batch_norm

        # flatten output of 2nd convolution layer
        self.layer2_out_shape = self.layer2_out.get_shape()                     # self.num_features = 3136 given above explicitly
        self.num_features = self.layer2_out_shape[1:4].num_elements()           # we need this number explicitly to define self.fc1_W above
        self.layer2_flat = tf.reshape(self.layer2_out, [-1, self.num_features]) # otherwise we have to define self.fc1_W right here

        # fully connected layers
        self.fc1_out = self.fully_connected(self.layer2_flat, self.fc1_W, self.fc1_b, relu=True, dropout=True)

        # logits and related
        self.logits = self.fully_connected(self.fc1_out, self.fc2_W, self.fc2_b, relu=False, dropout=False)
        self.y_pred = tf.nn.softmax(self.logits, name='y_pred')
        self.y_pred_cls = tf.cast(tf.argmax(self.y_pred, axis=1), tf.int32)

        # cost and optimizer
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                                                                        labels=self.y_true)
        self.cost = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost,
                                                                                           var_list=self.var_list)

        # cost_adversary and optimizer_adversary
        self.cost_adversary = self.cost + self.noise_weight * tf.nn.l2_loss(self.noise)
        self.optimizer_adversary = tf.train.AdamOptimizer(learning_rate=self.learning_rate_adversary).minimize(self.cost_adversary,
                                                                                                     var_list=self.var_list_adversary)

        # performance measure
        self.correct_bool = tf.equal(self.y_pred_cls, self.y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_bool, tf.float32))

    def train_adversary(self, adversarial_target_cls):
        self.adversarial_target_cls = adversarial_target_cls # construct y_batch
        y_batch = np.zeros((self.batch_size,10))             # to train noise
        y_batch[:,adversarial_target_cls] = 1                # so that CNN predict input images with added noise
        y_batch = y_batch.astype(np.float32)                 # as adversarial_target_cls

        for i in range(self.epoch_number_adversary):
            start_time = time.time() # start time of this epoch
            training_batch = zip(range(0, len(self.y_train), self.batch_size),
                                 range(self.batch_size, len(self.y_train), self.batch_size))
            idx_for_print_cost = 0
            for start, end in training_batch:                         # for each train
                feed_dict = {self.x: self.x_train[start:end,:],       # we pick up train batch sequentially
                             self.y_true: y_batch,                    # we don't use random batch
                             self.keep_prob: 1.0,                     # we use dropout with self.keep_prob: 0.5
                             self.is_train: False}                    # during train period
                self.sess.run(self.optimizer_adversary, feed_dict=feed_dict) # here is the main excution of train_adversary
                self.sess.run(self.noise_clip)                               # here is the main excution of train_adversary
                if idx_for_print_cost % 100 == 0:                               # for every 100 train
                    cost_adversary_now = self.sess.run(self.cost_adversary, feed_dict=feed_dict)  # we compute cost_adversary now
                    print(idx_for_print_cost, cost_adversary_now)                                 # we print cost_adversary now
                idx_for_print_cost += 1
            end_time = time.time() # end time of this epoch

            print("==========================================================")
            print("Epoch:", i)
            test_indices = np.arange(len(self.y_test))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:self.batch_size]
            feed_dict = {self.x: self.x_test[:self.batch_size,:],
                         self.y_true: self.y_test[:self.batch_size,:],
                         self.y_true_cls: self.y_test_cls[:self.batch_size],
                         self.keep_prob: 1.0, self.is_train: False}
            print("Accuracy on Random Test Samples: %g" % self.sess.run(self.accuracy, feed_dict=feed_dict))
            time_dif = end_time - start_time
            print("Time Usage: " + str(timedelta(seconds=int(round(time_dif)))))

    def train_adversary_random_batch(self, adversarial_target_cls):
        self.adversarial_target_cls = adversarial_target_cls  # construct y_batch
        y_batch = np.zeros((self.batch_size,10))              # to train noise
        y_batch[:,adversarial_target_cls] = 1                 # so that CNN predict input images with added noise
        y_batch = y_batch.astype(np.float32)                  # as adversarial_target_cls

        for i in range(self.num_iteration_adversary):
            idx = np.random.choice(self.x_train.shape[0], size=self.batch_size, replace=False)  # random_batch
            x_batch = self.x_train[idx]                                                         # random_batch
            feed_dict = {self.x: x_batch,
                         self.y_true: y_batch,
                         self.keep_prob: 1.0,
                         self.is_train: False}
            self.sess.run(self.optimizer_adversary, feed_dict=feed_dict)  # here is the main excution of train_adversary
            self.sess.run(self.noise_clip)                                # here is the main excution of train_adversary
            if (i % self.report_period == 0) or (i == self.num_iteration - 1):
                cost_adversary_now = self.sess.run(self.cost_adversary, feed_dict=feed_dict)                # we compute cost_adversary now
                logging.info('train iter : {:6d} | loss adversary : {:.6f}'.format(i, cost_adversary_now))  # we print cost_adversary now

    def plot_noise(self):
        image0 = self.sess.run(self.noise)
        utils.plot_one_image(image0, img_shape=self.img_shape)


class CNN_fashion_mnist(CNN_mnist):

    def data_loading(self, data):
        self.data = data

        # data pre-processing
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_train_cls, self.y_test_cls, self.class_names = self.data
        self.num_test = self.y_test.shape[0]

    def graph_construction(self):

        # graph dimension
        self.num_filters1 = 32   # Convolutional Layer 1: There are 32 of these filters.
        self.num_filters2 = 64   # Convolutional Layer 2: There are 64 of these filters.
        self.fc_size      = 1024 # Fully-connected layer: Number of neurons in fully-connected layer.
        self.filter_size1 = 5    # Convolutional Layer 1: Convolution filters are 5 x 5 pixels.
        self.filter_size2 = 5    # Convolutional Layer 2: Convolution filters are 5 x 5 pixels.
        self.num_features = 3136 # number of features in self.layer2_flat

        # data dimension
        self.img_size = 28
        self.img_shape = (28, 28)
        self.num_classes = 10
        self.num_channels = 1

        # placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.num_channels], name='x')
        self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')
        self.y_true_cls = tf.placeholder(tf.int32, shape=[None], name='y_true_cls')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.is_train = tf.placeholder(tf.bool, shape=[], name='is_train')

        # weights
        self.conv1_W = tf.get_variable("conv1_W",
                                       shape=(self.filter_size1, self.filter_size1, self.num_channels, self.num_filters1),
                                       #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.conv1_b = tf.get_variable("conv1_b",
                                       shape=(self.num_filters1, ),
                                       initializer=tf.constant_initializer(0.0))

        self.conv2_W = tf.get_variable("conv2_W",
                                       shape=(self.filter_size2, self.filter_size2, self.num_filters1, self.num_filters2),
                                       #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.conv2_b = tf.get_variable("conv2_b",
                                       shape=(self.num_filters2, ),
                                       initializer=tf.constant_initializer(0.0))

        self.fc1_W = tf.get_variable("fc1_W",
                                     shape=(self.num_features, self.fc_size),
                                     #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.fc1_b = tf.get_variable("fc1_b",
                                     shape=(self.fc_size, ),
                                     initializer=tf.constant_initializer(0.0))

        self.fc2_W = tf.get_variable("fc2_W",
                                     shape=(self.fc_size, self.num_classes),
                                     #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.fc2_b = tf.get_variable("fc2_b",
                                     shape=(self.num_classes, ),
                                     initializer=tf.constant_initializer(0.0))

        # convolution layers
        self.layer1_out = self.convolution(self.x, self.conv1_W, self.conv1_b)           # we don't use batch_norm
        self.layer2_out = self.convolution(self.layer1_out, self.conv2_W, self.conv2_b)  # we don't use batch_norm

        # flatten output of 2nd convolution layer
        self.layer2_out_shape = self.layer2_out.get_shape()                     # self.num_features = 3136 given above explicitly
        self.num_features = self.layer2_out_shape[1:4].num_elements()           # we need this number explicitly to define self.fc1_W above
        self.layer2_flat = tf.reshape(self.layer2_out, [-1, self.num_features]) # otherwise we have to define self.fc1_W right here

        # fully connected layers
        self.fc1_out = self.fully_connected(self.layer2_flat, self.fc1_W, self.fc1_b, relu=True, dropout=True)

        # logits and related
        self.logits = self.fully_connected(self.fc1_out, self.fc2_W, self.fc2_b, relu=False, dropout=False)
        self.y_pred = tf.nn.softmax(self.logits, name='y_pred')
        self.y_pred_cls = tf.cast(tf.argmax(self.logits, axis=1), tf.int32)

        # cost and optimizer
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                                                                        labels=self.y_true)
        self.cost = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # performance measure
        self.correct_bool = tf.equal(self.y_pred_cls, self.y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_bool, tf.float32))




class CNN_fashion_mnist_hyperparameter_search(CNN_mnist):

    def __init__(self, sess, batch_size=128, report_period=100, learning_rate=1e-3,
                 epoch_number=2,           # this is used when we train without using random batch
                 num_iteration=int(1e+3), # this is used when we train with using random batch
                 fc_size=int(1024)):

        self.sess = sess
        self.batch_size = int(batch_size)
        self.report_period = int(report_period)
        self.learning_rate = float(learning_rate)
        self.epoch_number = int(epoch_number)
        self.num_iteration = int(num_iteration)
        self.fc_size = int(fc_size)

    def data_loading(self, data):
        self.data = data

        # data pre-processing
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_train_cls, self.y_test_cls, self.class_names = self.data
        self.num_test = self.y_test.shape[0]

    def graph_construction(self):

        # graph dimension
        self.num_filters1 = 32   # Convolutional Layer 1: There are 32 of these filters.
        self.num_filters2 = 64   # Convolutional Layer 2: There are 64 of these filters.
        # self.fc_size      = 1024 # Fully-connected layer: Number of neurons in fully-connected layer.
        self.filter_size1 = 5    # Convolutional Layer 1: Convolution filters are 5 x 5 pixels.
        self.filter_size2 = 5    # Convolutional Layer 2: Convolution filters are 5 x 5 pixels.
        self.num_features = 3136 # number of features in self.layer2_flat

        # data dimension
        self.img_size = 28
        self.img_shape = (28, 28)
        self.num_classes = 10
        self.num_channels = 1

        # placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.num_channels], name='x')
        self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')
        self.y_true_cls = tf.placeholder(tf.int32, shape=[None], name='y_true_cls')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.is_train = tf.placeholder(tf.bool, shape=[], name='is_train')

        # weights
        self.conv1_W = tf.get_variable("conv1_W",
                                       shape=(self.filter_size1, self.filter_size1, self.num_channels, self.num_filters1),
                                       #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.conv1_b = tf.get_variable("conv1_b",
                                       shape=(self.num_filters1, ),
                                       initializer=tf.constant_initializer(0.0))

        self.conv2_W = tf.get_variable("conv2_W",
                                       shape=(self.filter_size2, self.filter_size2, self.num_filters1, self.num_filters2),
                                       #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.conv2_b = tf.get_variable("conv2_b",
                                       shape=(self.num_filters2, ),
                                       initializer=tf.constant_initializer(0.0))

        self.fc1_W = tf.get_variable("fc1_W",
                                     shape=(self.num_features, self.fc_size),
                                     #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.fc1_b = tf.get_variable("fc1_b",
                                     shape=(self.fc_size, ),
                                     initializer=tf.constant_initializer(0.0))

        self.fc2_W = tf.get_variable("fc2_W",
                                     shape=(self.fc_size, self.num_classes),
                                     #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.fc2_b = tf.get_variable("fc2_b",
                                     shape=(self.num_classes, ),
                                     initializer=tf.constant_initializer(0.0))

        # convolution layers
        self.layer1_out = self.convolution(self.x, self.conv1_W, self.conv1_b)           # we don't use batch_norm
        self.layer2_out = self.convolution(self.layer1_out, self.conv2_W, self.conv2_b)  # we don't use batch_norm

        # flatten output of 2nd convolution layer
        self.layer2_out_shape = self.layer2_out.get_shape()                     # self.num_features = 3136 given above explicitly
        self.num_features = self.layer2_out_shape[1:4].num_elements()           # we need this number explicitly to define self.fc1_W above
        self.layer2_flat = tf.reshape(self.layer2_out, [-1, self.num_features]) # otherwise we have to define self.fc1_W right here

        # fully connected layers
        self.fc1_out = self.fully_connected(self.layer2_flat, self.fc1_W, self.fc1_b, relu=True, dropout=True)

        # logits and related
        self.logits = self.fully_connected(self.fc1_out, self.fc2_W, self.fc2_b, relu=False, dropout=False)
        self.y_pred = tf.nn.softmax(self.logits, name='y_pred')
        self.y_pred_cls = tf.cast(tf.argmax(self.logits, axis=1), tf.int32)

        # cost and optimizer
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                                                                        labels=self.y_true)
        self.cost = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # performance measure
        self.correct_bool = tf.equal(self.y_pred_cls, self.y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_bool, tf.float32))





































class CNN_cifar10(CNN_mnist):

    def data_loading(self, data):
        self.data = data

        # data pre-processing
        self.x_train, self.y_train_cls, self.y_train, self.x_test, self.y_test_cls, self.y_test, self.class_names = self.data
        self.num_test = self.y_test.shape[0]

    def graph_construction(self):

        # graph dimension
        self.num_filters1 = 32   # Convolutional Layer 1: There are 32 of these filters.
        self.num_filters2 = 64   # Convolutional Layer 2: There are 64 of these filters.
        self.fc_size      = 1024 # Fully-connected layer: Number of neurons in fully-connected layer.
        self.filter_size1 = 5    # Convolutional Layer 1: Convolution filters are 5 x 5 pixels.
        self.filter_size2 = 5    # Convolutional Layer 2: Convolution filters are 5 x 5 pixels.
        self.num_features = 4096 # number of features in self.layer2_flat

        # data dimension
        self.img_size = 32
        self.img_shape = (32, 32, 3)
        self.num_classes = 10
        self.num_channels = 3

        # placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.num_channels], name='x')
        self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')
        self.y_true_cls = tf.placeholder(tf.int32, shape=[None], name='y_true_cls')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.is_train = tf.placeholder(tf.bool, shape=[], name='is_train')

        # weights
        self.conv1_W = tf.get_variable("conv1_W",
                                       shape=(self.filter_size1, self.filter_size1, self.num_channels, self.num_filters1),
                                       #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.conv1_b = tf.get_variable("conv1_b",
                                       shape=(self.num_filters1, ),
                                       initializer=tf.constant_initializer(0.0))

        self.conv2_W = tf.get_variable("conv2_W",
                                       shape=(self.filter_size2, self.filter_size2, self.num_filters1, self.num_filters2),
                                       #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.conv2_b = tf.get_variable("conv2_b",
                                       shape=(self.num_filters2, ),
                                       initializer=tf.constant_initializer(0.0))

        self.fc1_W = tf.get_variable("fc1_W",
                                     shape=(self.num_features, self.fc_size),
                                     #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.fc1_b = tf.get_variable("fc1_b",
                                     shape=(self.fc_size, ),
                                     initializer=tf.constant_initializer(0.0))

        self.fc2_W = tf.get_variable("fc2_W",
                                     shape=(self.fc_size, self.num_classes),
                                     #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.fc2_b = tf.get_variable("fc2_b",
                                     shape=(self.num_classes, ),
                                     initializer=tf.constant_initializer(0.0))

        # convolution layers
        self.layer1_out = self.convolution(self.x, self.conv1_W, self.conv1_b)           # we don't use batch_norm
        self.layer2_out = self.convolution(self.layer1_out, self.conv2_W, self.conv2_b)  # we don't use batch_norm

        # flatten output of 2nd convolution layer
        self.layer2_out_shape = self.layer2_out.get_shape()                     # self.num_features = 4096 given above explicitly
        self.num_features = self.layer2_out_shape[1:4].num_elements()           # we need this number explicitly to define self.fc1_W above
        self.layer2_flat = tf.reshape(self.layer2_out, [-1, self.num_features]) # otherwise we have to define self.fc1_W right here

        # fully connected layers
        self.fc1_out = self.fully_connected(self.layer2_flat, self.fc1_W, self.fc1_b, relu=True, dropout=True)

        # logits and related
        self.logits = self.fully_connected(self.fc1_out, self.fc2_W, self.fc2_b, relu=False, dropout=False)
        self.y_pred = tf.nn.softmax(self.logits, name='y_pred')
        self.y_pred_cls = tf.cast(tf.argmax(self.y_pred, axis=1), tf.int32)

        # cost and optimizer
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                                                                        labels=self.y_true)
        self.cost = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # performance measure
        self.correct_bool = tf.equal(self.y_pred_cls, self.y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_bool, tf.float32))


class CNN_cifar10_random_crop(CNN_cifar10):

    def pre_process_image(self,
        image,      # This function takes a single image as input,
        is_train    # and a placeholder for a boolean whether to build the training or testing graph.
        ):

        if is_train is not None:
            # For training, add the following to the TensorFlow graph.

            # Randomly crop the input image.
            image = tf.random_crop(image, size=[self.img_size_cropped, self.img_size_cropped, self.num_channels])

            # Randomly flip the image horizontally.
            image = tf.image.random_flip_left_right(image)

            # Randomly adjust hue, contrast and saturation.
            image = tf.image.random_hue(image, max_delta=0.05)
            image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

            # Some of these functions may overflow and result in pixel
            # values beyond the [0, 1] range. It is unclear from the
            # documentation of TensorFlow 0.10.0rc0 whether this is
            # intended. A simple solution is to limit the range.

            # Limit the image pixels between [0, 1] in case of overflow.
            image = tf.minimum(image, 1.0)
            image = tf.maximum(image, 0.0)
        else:
            # For not training, i.e., test, add the following to the TensorFlow graph.

            # Crop the input image around the centre so it is the same
            # size as images that are randomly cropped during training.
            image = tf.image.resize_image_with_crop_or_pad(image,
                                                           target_height=img_size_cropped,
                                                           target_width=img_size_cropped)

        return image

    def pre_process_images(self, images, is_train):
        # Use TensorFlow to loop over all the input images and call
        # the function above which takes a single image as input.
        images = tf.map_fn(lambda image: self.pre_process_image(image, is_train), images)

        return images

    def graph_construction(self):

        # graph dimension
        self.num_filters1 = 32   # Convolutional Layer 1: There are 32 of these filters.
        self.num_filters2 = 64   # Convolutional Layer 2: There are 64 of these filters.
        self.fc_size      = 1024 # Fully-connected layer: Number of neurons in fully-connected layer.
        self.filter_size1 = 5    # Convolutional Layer 1: Convolution filters are 5 x 5 pixels.
        self.filter_size2 = 5    # Convolutional Layer 2: Convolution filters are 5 x 5 pixels.
        self.num_features = 2304 # number of features in self.layer2_flat

        # data dimension
        self.img_size = 32
        self.img_shape = (32, 32, 3)
        self.num_classes = 10
        self.num_channels = 3

        # cropped data dimension
        self.img_size_cropped = 24
        self.img_shape_cropped = (24, 24, 3)

        # placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.num_channels], name='x')
        self.y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')
        self.y_true_cls = tf.placeholder(tf.int32, shape=[None], name='y_true_cls')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.is_train = tf.placeholder(tf.bool, shape=[], name='is_train')

        # crop original images self.x by random cropping during train and center cropping during test
        self.x_cropped = tf.identity(self.pre_process_images(images=self.x, is_train=self.is_train), name='x_distorted')

        # weights
        self.conv1_W = tf.get_variable("conv1_W",
                                       shape=(self.filter_size1, self.filter_size1, self.num_channels, self.num_filters1),
                                       #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.conv1_b = tf.get_variable("conv1_b",
                                       shape=(self.num_filters1, ),
                                       initializer=tf.constant_initializer(0.0))

        self.conv2_W = tf.get_variable("conv2_W",
                                       shape=(self.filter_size2, self.filter_size2, self.num_filters1, self.num_filters2),
                                       #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.conv2_b = tf.get_variable("conv2_b",
                                       shape=(self.num_filters2, ),
                                       initializer=tf.constant_initializer(0.0))

        self.fc1_W = tf.get_variable("fc1_W",
                                     shape=(self.num_features, self.fc_size),
                                     #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.fc1_b = tf.get_variable("fc1_b",
                                     shape=(self.fc_size, ),
                                     initializer=tf.constant_initializer(0.0))

        self.fc2_W = tf.get_variable("fc2_W",
                                     shape=(self.fc_size, self.num_classes),
                                     #initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"))
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.fc2_b = tf.get_variable("fc2_b",
                                     shape=(self.num_classes, ),
                                     initializer=tf.constant_initializer(0.0))

        # convolution layers
        self.layer1_out = self.convolution(self.x_cropped, self.conv1_W, self.conv1_b)   # we don't use batch_norm
        self.layer2_out = self.convolution(self.layer1_out, self.conv2_W, self.conv2_b)  # we don't use batch_norm

        # flatten output of 2nd convolution layer
        self.layer2_out_shape = self.layer2_out.get_shape()                     # self.num_features = 3136 given above explicitly
        self.num_features = self.layer2_out_shape[1:4].num_elements()           # we need this number explicitly to define self.fc1_W above
        self.layer2_flat = tf.reshape(self.layer2_out, [-1, self.num_features]) # otherwise we have to define self.fc1_W right here

        # fully connected layers
        self.fc1_out = self.fully_connected(self.layer2_flat, self.fc1_W, self.fc1_b, relu=True, dropout=True)

        # logits and related
        self.logits = self.fully_connected(self.fc1_out, self.fc2_W, self.fc2_b, relu=False, dropout=False)
        self.y_pred = tf.nn.softmax(self.logits, name='y_pred')
        self.y_pred_cls = tf.cast(tf.argmax(self.y_pred, axis=1), tf.int32)

        # cost and optimizer
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                                                                        labels=self.y_true)
        self.cost = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # performance measure
        self.correct_bool = tf.equal(self.y_pred_cls, self.y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_bool, tf.float32))

    def train(self):
        for i in range(self.epoch_number):
            start_time = time.time() # start time of this epoch
            training_batch = zip(range(0, len(self.y_train), self.batch_size),
                                 range(self.batch_size, len(self.y_train), self.batch_size))
            idx_for_print_cost = 0
            for start, end in training_batch:                         # for each train
                feed_dict = {self.x: self.x_train[start:end,:],       # we pick up train batch sequentially
                             self.y_true: self.y_train[start:end,:],  # we don't use random batch
                             self.keep_prob: 0.5,                     # we use dropout with self.keep_prob: 0.5
                             self.is_train: True}                     # during train period
                self.sess.run(self.optimizer, feed_dict=feed_dict)    # here is the main excution of train
                if idx_for_print_cost % self.report_period == 0:              # for every self.report_period train
                    cost_now = self.sess.run(self.cost, feed_dict=feed_dict)  # we compute cost now
                    print(idx_for_print_cost, cost_now)                       # we print cost now
                idx_for_print_cost += 1
            end_time = time.time() # end time of this epoch

            print("==========================================================")
            print("Epoch:", i)
            test_indices = np.arange(len(self.y_test))                                                       # at the end of each epoch
            np.random.shuffle(test_indices)                                                                  # we choose random batch from test set
            test_indices = test_indices[0:self.batch_size]                                                   # and compute accuracy on this random test set
            feed_dict = {self.x: self.x_test[:self.batch_size,:],                                            #
                         self.y_true: self.y_test[:self.batch_size,:],                                       #
                         self.y_true_cls: self.y_test_cls[:self.batch_size],                                 #
                         self.keep_prob: 1.0,                                                                #
                         self.is_train: False}                                                               #
            print("Accuracy on Random Test Samples: %g" % self.sess.run(self.accuracy, feed_dict=feed_dict)) #
            time_dif = end_time - start_time                                     # we check computing time for each epoch
            print("Time Usage: " + str(timedelta(seconds=int(round(time_dif))))) # and print it

    def train_random_batch(self):
        for i in range(self.num_iteration):
            idx = np.random.choice(self.x_train.shape[0], size=self.batch_size, replace=False) # random_batch
            x_batch, y_batch = self.x_train[idx], self.y_train[idx]                            # random_batch
            feed_dict = {self.x: x_batch,
                         self.y_true: y_batch,
                         self.keep_prob: 0.5,
                         self.is_train: True}
            self.sess.run(self.optimizer, feed_dict=feed_dict)
            if (i % self.report_period == 0) or (i == self.num_iteration - 1):
                loss = self.sess.run(self.cost, feed_dict=feed_dict)
                logging.info('train iter : {:6d} | loss : {:.6f}'.format(i, loss))

