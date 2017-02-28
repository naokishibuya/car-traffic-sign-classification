"""
A wrapper class for tf.Session
"""
import os
import tensorflow as tf


class Session:
    """
    A wrapper class for tf.Session
    """
    def __init__(self):
        """
        Clean up the default graph first and then create a new session
        """
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)


    def load(self, path):
        """
        Load a session state from the checkpoint file
        """
        saver = tf.train.Saver()
        saver.restore(self.sess, path)


    def save(self, path):
        """
        Save the session state into a checkpoint file
        """
        save_dir = os.path.dirname(path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        saver = tf.train.Saver()
        saver.save(self.sess, path)


    def close(self):
        """
        Close the session
        """
        if self.sess is not None:
            self.sess.close()
            self.sess = None


    def __enter__(self):
        """
        For the beginning of the 'with' statement
        """
        return self


    def __exit__(self, type, value, traceback):
        """
        For the end of the 'with' statement
        """
        self.close()
