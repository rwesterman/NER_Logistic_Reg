# adagrad_trainer.py

import utils
from abc import ABC, abstractmethod
import numpy as np


class LogisticLoss():
    # Create this after the indexer and pass it an instance?
    def __init__(self, indexer):

        # indexer is an Indexer object with features initialized. This is mostly used to get correct length for weight vector
        self.indexer = indexer

    def update_indexer(self, indexer):
        self.indexer = indexer

    def features_to_index(self, feat_list):
        """
        :param feat_list: list of strings representing features of a single token (first element is token string itself)
        :return: list of integers representing the index of applicable features
        """
        feat_index = []
        for i, feature in enumerate(feat_list):
            # Don't need to skip token value because it won't show up in indexer and therefore won't get added
            feat_index = utils.maybe_add_feature(feat_index, self.indexer, False, feature)

        return feat_index

    def get_scores(self, feat_index, weights):
        # Todo: Fill this in based on get_scores_from_features function
        """
        runs score_indexed_features
        :return: score calculated from feat_index and weights
        """
        score = utils.score_indexed_features(feat_index, weights)
        return score

    def sigmoid(self, feat_list, weights):
        """Implement logistic regression here. Takes two numpy arrays, calculates their dot product,
            and plugs it into sigmoid formula"""
        # z = np.dot(weights, inputs)
        feat_index = self.features_to_index(feat_list)
        z = self.get_scores(feat_index, weights)
        sig_val = np.exp(z) / (1 + np.exp(z))
        return sig_val


    # takes in label value, list of features, and weights. Calculates and returns gradient
    def calculate_gradient(self, label, token_feat_list, weights):
        # feat_vect does not hold any zero values, but instead it is used to index
        # which features are applicable to a particular word. ie feat_vec == [4,5] means
        # the 4th and 5th features are 1, and all others are 0

        sig_val = self.sigmoid(token_feat_list, weights)
        feat_index = self.features_to_index(token_feat_list)

        # initialize gradient as a vector the same size as your feature list, and all zero values. Values will be changed where appropraite
        grad = np.zeros(len(self.indexer))

        # The cost function being used is (sigma(wx) - y) where sigma is the logistic value, and y is the correct label
        # This function keeps all values zero except those that have features show up
        for index_val in feat_index:
            grad[index_val] = (sig_val - label)

        return grad

class Optimizer(ABC):
    # Scores a sparse feature vector
    # feats: list of integer feature indices
    def score(self, feats):
        i = 0
        score = 0.0
        while i < len(feats):
            score += self.access(feats[i])
            i += 1
        return score

    @abstractmethod
    def apply_gradient_update(self, gradient, batch_size):
        pass

    @abstractmethod
    def access(self, i):
        pass

    @abstractmethod
    def get_final_weights(self, i):
        pass


# SGD optimizer implementation, designed to have the same interface as the Adagrad optimizers
# Weights are stores in the optimizer for training.
class SGDOptimizer(Optimizer):
    # Todo: Alter this to MINIMIZE LOSS rather than MAXIMIZING LOG LIKELIHOOD
    # init_weights: a numpy array of the correct dimension, usually initialized to 0
    # alpha: step size
    def __init__(self, init_weights, alpha=0.1):
        self.weights = init_weights
        self.alpha = alpha

    # gradient is changed to just be the vector by which to change the weights
    def apply_gradient_update(self, gradient, batch_size):
        # originally gradient was a Counter() object. I am instead calculating it in
        # my LogisticLoss object and sending it here for the simple implementation.
        self.weights = self.weights - self.alpha * gradient

        # for i in gradient.keys():
        #     g = gradient.get_count(i)
        #     # self.weights[i] = self.weights[i] + self.alpha * g
        #     self.weights[i] = self.weights[i] - self.alpha * g


    # Get the weight of feature i
    def access(self, i):
        return self.weights[i]

    def get_final_weights(self):
        return self.weights


# Wraps a weight vector and applies the Adagrad update using second moments of features to make custom step sizes.
# This version incorporates L1 regularization: while this regularization should be applied to squash the feature vector
# on every gradient update, we instead evaluate the regularizer lazily only when the particular feature is touched
# (either by gradient update or by access). approximate lets you turn this off for faster access, but regularization is
# now applied somewhat inconsistently.
# See section 5.1 of http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf for more details
class L1RegularizedAdagradTrainer(Optimizer):
    # init_weights: a numpy array of the correct dimension, usually initialized to 0
    # lamb: float lambda constant for the regularizer. Values above 0.01 will often cause all features to be zeroed out.
    # eta: float step size. Values from 0.01 to 10 often work well.
    # approximate: turns off gradient updates on access, only uses them when weights are written to.
    # So regularization is applied inconsistently, but it makes things faster.
    def __init__(self, init_weights, lamb=1e-8, eta=1.0, use_regularization=False, approximate=True):
        self.weights = init_weights
        self.lamb = lamb
        self.eta = eta
        self.use_regularization = use_regularization
        self.approximate = approximate
        self.curr_iter = 0
        self.last_iter_touched = [0 for i in range(0, self.weights.shape[0])]
        self.diag_Gt = np.zeros_like(self.weights, dtype=float)

    # Take a sparse representation of the gradient and make an update, normalizing by the batch size to keep
    # hyperparameters constant as the batch size is varied
    # gradient: Counter
    # batch_size: integer
    def apply_gradient_update(self, gradient, batch_size):
        batch_size_multiplier = 1.0 / batch_size
        self.curr_iter += 1
        for i in gradient.keys():
            xti = self.weights[i]
            # N.B.We negate the gradient here because the Adagrad formulas are all for minimizing
            # and we're trying to maximize, so think of it as minimizing the negative of the objective
            # which has the opposite gradient
            # See section 5.1 in http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf for more details
            # eta is the step size, lambda is the regularization
            gti = -gradient.get_count(i) * batch_size_multiplier
            old_eta_over_Htii = self.eta / (1 + np.sqrt(self.diag_Gt[i]))
            self.diag_Gt[i] += gti * gti
            Htii = 1 + np.sqrt(self.diag_Gt[i])
            eta_over_Htii = self.eta / Htii
            new_xti = xti - eta_over_Htii * gti
            # Apply the regularizer for every iteration since touched
            iters_since_touched = self.curr_iter - self.last_iter_touched[i]
            self.last_iter_touched[i] = self.curr_iter
            self.weights[i] = np.sign(new_xti) * max(0, np.abs(new_xti) - self.lamb * eta_over_Htii - (iters_since_touched - 1) * self.lamb * old_eta_over_Htii)

    # Get the weight of feature i
    def access(self, i):
        if not self.approximate and self.last_iter_touched[i] != self.curr_iter:
            xti = self.weights[i]
            Htii = 1 + np.sqrt(self.diag_Gt[i])
            eta_over_Htii = self.eta / Htii
            iters_since_touched = self.curr_iter - self.last_iter_touched[i]
            self.last_iter_touched[i] = self.curr_iter
            self.weights[i] = np.sign(xti) * max(0, np.abs(xti) - iters_since_touched * self.lamb * self.eta * eta_over_Htii);
        return self.weights[i]

    # Return a numpy array containing the final weight vector values -- manually calls access to force each weight to
    # have an updated value.
    def get_final_weights(self):
        for i in range(0, self.weights.shape[0]):
            self.access(i)
        return self.weights


# Applies the Adagrad update with no regularization. Will be substantially faster than the L1 regularized version
# due to less computation required to update each feature.
class UnregularizedAdagradTrainer(Optimizer):
    def __init__(self, init_weights, eta=1.0):
        self.weights = init_weights
        self.eta = eta
        self.diag_Gt = np.zeros_like(self.weights, dtype=float)

    def apply_gradient_update(self, gradient, batch_size):
        batch_size_multiplier = 1.0 / batch_size
        for i in gradient.keys():
            xti = self.weights[i]
            gti = -gradient.get_count(i) * batch_size_multiplier
            self.diag_Gt[i] += gti * gti
            Htii = 1 + np.sqrt(self.diag_Gt[i])
            eta_over_Htii = self.eta / Htii
            self.weights[i] = xti - eta_over_Htii * gti

    # Get the weight of feature i
    def access(self, i):
        return self.weights[i]

    # Return a numpy array containing the final weight vector values -- manually calls access to force each weight to
    # have an updated value.
    def get_final_weights(self):
        return self.weights
