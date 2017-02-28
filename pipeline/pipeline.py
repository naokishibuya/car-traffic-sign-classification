"""
Adapters for Scikit-Learn's Pipeline framework
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from .network import NeuralNetworkClassifier


class Transformer(BaseEstimator, TransformerMixin):
    """
    A generic transformer class that uses a given function to transform inputs
    """
    def __init__(self, func):
        self.func = func


    def fit(self, X, y=None):
        """
        This is a static transformer so we do nothing here
        """
        return self


    def transform(self, X):
        """
        Apply the function here
        """
        return np.array([self.func(x) for x in X])


class Estimator(BaseEstimator, ClassifierMixin):
    """
    Generic classifier class that uses a given model to handle estimator operations
    """
    def __init__(self, model, batch_size, k=5): # adjust the batch_size for your computer memory, GPU
        self.model = model
        self.batch_size = batch_size
        self.k = k


    def fit(self, X, y):
        """
        Required by scikit-learn
        """
        for start in range(0, len(X), self.batch_size):
            end = start + self.batch_size
            self.model.train(X[start:end], y[start:end])
        return self


    def predict(self, X, y=None):
        """
        Required by scikit-learn
        """
        pred = []
        for start in range(0, len(X), self.batch_size):
            end = start + self.batch_size
            p = self.model.predict(X[start:end])
            pred.extend(p)
        return pred


    def predict_proba(self, X):
        """
        Returns probability instead of prediction.
        """
        prob = []   # top probability
        self.top_k_ = [] # top k predictions and probabilities
        for start in range(0, len(X), self.batch_size):
            end = start + self.batch_size
            pb, tk = self.model.predict_proba(X[start:end], k=self.k)
            prob.extend(pb)
            self.top_k_.extend(tk)
        return prob


def build_pipeline(functions, session, network, optimizer=None, batch_size=20):
    """
    Combine a list of transformers and an estimator to make a new pipeline
    """
    transformers = [Transformer(func) for func in functions]
    classifier = NeuralNetworkClassifier(session, network, optimizer)
    estimator = Estimator(classifier, batch_size)
    return make_pipeline(*transformers, estimator)
