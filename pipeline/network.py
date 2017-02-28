"""
A wrapper class for tensorflow primitives

Examples
--------
>>> with Session() as session:
>>>     # construct the network
>>>     network = make_network1()
>>>     optimizer = make_adam(1.0e-3)
>>>     nnc = NeuralNetworkClassifier(session, network, optimizer)
>>>
>>>     # restore the saved parameter values into the network
>>>     session.load('checkpoint/mymodel1.ckpt')
>>>
>>>     # further training
>>>     nnc.train(...features..., ...labels...)
>>>     # save the current session state
>>>     session.save('checkpoint/mymodel2.ckpt')
>>>
>>>     # more training
>>>     nnc.train(...features..., ...labels...)
>>>     #... do something more and save another ...
>>>     session.save('checkpoint/mymodel3.ckpt')
>>>
>>>     ...
>>>
>>>     # session will be automatically closed
"""
import tensorflow as tf
from tensorflow.contrib.layers import flatten


class NeuralNetwork:
    """
    A wrapper class for tensorflow primitives
    """
    def __init__(self, weight_mu=0, weight_sigma=0.1):
        self.weight_mu = weight_mu
        self.weight_sigma = weight_sigma
        self.layer = 0
        self.z = None


    def variable_scope(self, name=None):
        """
        Provide a name scope
        """
        if name is not None:
            self.layer += 1
            self.layer_name = '{}_{}'.format(self.layer, name)
        return tf.variable_scope(self.layer_name)


    def input(self, shape):
        """
        Add an input layer
        """
        with self.variable_scope('input'):
            self.x = tf.placeholder(tf.float32, (None, *shape), name='x')
            self.y = tf.placeholder(tf.int64, (None), name='y')
            self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.z = self.x
        return self


    def W_b(self, shape):
        """
        Weight and bias initialization
        """
        W = tf.Variable(tf.truncated_normal(shape=shape,
                                            mean=self.weight_mu,
                                            stddev=self.weight_sigma), name='W')
        b = tf.Variable(tf.zeros(shape[-1]), name='b')
        return W, b


    def conv(self, shape, strides=(1, 1, 1, 1), padding='VALID'):
        """
        Add a convolutional layer
        """
        with self.variable_scope('conv'):
            shape = (shape[0], shape[1], int(self.z.get_shape()[-1]), shape[2])
            W, b = self.W_b(shape)
            self.z = tf.nn.conv2d(self.z, W, strides=strides, padding=padding) + b
        return self


    def dense(self, size):
        """
        Add a dense (fully connected) layer
        """
        with self.variable_scope('dense'):
            shape = [self.z.get_shape().as_list()[-1], size]
            W, b = self.W_b(shape)
            self.z = tf.matmul(self.z, W) + b
        return self


    def relu(self, leak_ratio=0.0):
        """
        Add a ReLU activation
        """
        with self.variable_scope():
            if leak_ratio > 0.0:
                self.z = tf.maximum(self.z, self.z*leak_ratio)
            else:
                self.z = tf.nn.relu(self.z)
        return self


    def elu(self):
        """
        Add a ELU activation
        """
        with self.variable_scope():
            self.z = tf.nn.elu(self.z)
        return self


    def max_pool(self, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID'):
        """
        Add a max pooling
        """
        with self.variable_scope():
            self.z = tf.nn.max_pool(self.z, ksize, strides, padding)
        return self


    def dropout(self, keep_prob):
        """
        Add a drop out
        """
        with self.variable_scope():
            self.z = tf.cond(self.is_training,
                             lambda: tf.nn.dropout(self.z, keep_prob),
                             lambda: self.z)
        return self


    def flatten(self):
        """
        Flatten the dimension
        """
        with self.variable_scope():
            self.z = flatten(self.z)
        return self


class NeuralNetworkClassifier:
    """
    A classifier that uses the Network object inside.
    """
    def __init__(self, session, network, optimizer=None):
        self.sess = session.sess
        self.x = network.x # features
        self.y = network.y # labels
        self.z = network.z # logits
        self.is_training = network.is_training
        self.init(optimizer)


    def init(self, optimizer):
        """
        Build optimization and prediction operators
        """
        with tf.variable_scope('evaluation'):
            one_hot_y = tf.one_hot(self.y, self.z.get_shape()[-1], name='one_hot_y')
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.z,
                labels=one_hot_y), name='loss')

        if optimizer:
            with tf.variable_scope('training'):
                self.training = optimizer.minimize(loss)

        with tf.variable_scope('prediction'):
            self.prediction = tf.argmax(self.z, 1, name='prediction')
            self.probability = tf.nn.softmax(self.z, name='probability')
            self.k = tf.placeholder(tf.int32, name='k')
            self.top_k = tf.nn.top_k(self.probability, k=self.k)

        self.sess.run(tf.global_variables_initializer())


    def train(self, x, y):
        """
        Perform the training
        """
        self.sess.run(self.training, feed_dict={self.x: x, self.y: y, self.is_training: True})


    def predict(self, x):
        """
        Perform the prediction
        """
        return self.sess.run(self.prediction, feed_dict={self.x: x, self.is_training: False})


    def predict_proba(self, x, k=None):
        """
        Return the probabilities.  If k is given, returns the top k probabilities.
        """
        return self.sess.run([self.probability, self.top_k],
                             feed_dict={self.x: x, self.is_training: False, self.k: k})


def make_adam(learning_rate):
    """
    A helper to create adam optimizer
    """
    return tf.train.AdamOptimizer(learning_rate=learning_rate)
