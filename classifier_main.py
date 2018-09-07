# classifier_main.py

import argparse
import sys
import time
from nerdata import *
from utils import *
from optimizers import *
import re
from collections import defaultdict
from features import *
from pprint import pprint

import numpy as np

# Command-line arguments to the system -- you can extend these if you want, but you shouldn't need to modify any of them
def _parse_args():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='BAD', help='model to run (BAD, CLASSIFIER)')
    parser.add_argument('--train_path', type=str, default='data/eng.train', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/eng.testa', help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default='data/eng.testb.blind', help='path to dev set (you should not need to modify)')
    parser.add_argument('--test_output_path', type=str, default='eng.testb.out', help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing output on the test set')
    args = parser.parse_args()
    return args

# Wrapper for an example of the person binary classification task.
# tokens: list of string words
# labels: list of (0, 1) where 0 is non-name, 1 is name
class PersonExample(object):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)


# Changes NER-style chunk examples into binary classification examples.
def transform_for_classification(ner_exs):
    # Take each LabeledSentence object and extract the bio tags.
    for labeled_sent in ner_exs:
        tags = bio_tags_from_chunks(labeled_sent.chunks, len(labeled_sent))

        # create a list "labels" that has 1 for every position with a person's name, 0 otherwise
        labels = [1 if tag.endswith("PER") else 0 for tag in tags]

        # Yield a PersonExample object with a list of tokens and labels
        yield PersonExample([tok.word for tok in labeled_sent.tokens], labels)

# Person classifier that takes counts of how often a word was observed to be the positive and negative class
# in training, and classifies as positive any tokens which are observed to be positive more than negative.
# Unknown tokens or ties default to negative.
class CountBasedPersonClassifier(object):
    def __init__(self, pos_counts, neg_counts):
        self.pos_counts = pos_counts
        self.neg_counts = neg_counts

    def predict(self, tokens, idx):
        # simply checks that the "positive" word count is higher than the "negative" word count in a sentence
        if self.pos_counts.get_count(tokens[idx]) > self.neg_counts.get_count(tokens[idx]):
            return 1
        else:
            return 0

# "Trains" the count-based person classifier by collecting counts over the given examples.
def train_count_based_binary_classifier(ner_exs):
    pos_counts = Counter()
    neg_counts = Counter()
    for ex in ner_exs:
        for idx in range(0, len(ex)):
            if ex.labels[idx] == 1:
                pos_counts.increment_count(ex.tokens[idx], 1.0)
            else:
                neg_counts.increment_count(ex.tokens[idx], 1.0)
    return CountBasedPersonClassifier(pos_counts, neg_counts)


# "Real" classifier that takes a weight vector
# This will be used in the actual evaluation but not in the training
class PersonClassifier(object):
    def __init__(self, weights, indexer):
        self.weights = weights
        self.indexer = indexer
        self.loss = LogisticLoss(self.indexer)
        self.stop_words = get_stop_words()
        self.tokens = []
        self.feature_dict = {}

    # Makes a prediction for token at position idx in the given PersonExample
    def predict(self, tokens, idx):
        # Get the features for the full sentence.
        # Todo: Look into modifying this if it takes too much time??
        # Only create features for the sentence when the first token is accessed
        # if idx == 0:
        if self.tokens != tokens:
            self.feature_dict = get_applicable_feats(tokens, self.stop_words, indexer)
            self.tokens = tokens

        # Get the sigmoid predictor using the feature_dict and the final weights
        value = self.loss.sigmoid(self.feature_dict[idx], self.weights)
        if value > 0.5:
            return 1
        else: return 0

def get_stop_words():
    stop_words = []
    with open("stop_words.txt", "r") as f:
        for line in f:
            stop = f.readline()
            stop = stop.strip()
            if stop != "":
                stop_words.append(stop)
    return stop_words

def train_classifier(ner_exs):
    # Create an Indexer object to track features, then initialize it with feature set
    indexer = Indexer()
    indexer = init_features(indexer)

    # initialize SGDoptimizer with random weights and 0.1 learning rate
    sgd = SGDOptimizer(np.random.rand(len(indexer)), 0.1)

    # initialize a LogisticLoss object. Will need to send it the
    loss = LogisticLoss(indexer)

    stop_words = get_stop_words()

    # Do all featurization here, before the training loops...
    # Want a huge list of lists with features for every token


    for epochs in range(25):

        for sent_ex in ner_exs:
            labels = sent_ex.labels
            tokens = sent_ex.tokens
            # Get the applicable features for each word in a sentence.
            # important to do this at sentence level because some features depend on other words in sentence
            feature_dict, indexer = get_applicable_feats(tokens, stop_words, indexer)

            # update the indexer in the LogisticLoss class so that len(indexer) stays consistent
            loss.update_indexer(indexer)

            # feature_dict will have form {0:["token", "feat1", "feat5"], 1:...}
            # where keys are the position of the word in a sentence, and the first value in list is the token itself

            # for each word in the sentence
            for key, feat_list in feature_dict.items():
                # get current weights
                weights = sgd.weights

                #labels[key] is the correct label for each token, since the key is the position of the
                # token in the sentence. feat_list is the list of features (in strings) for the given token
                gradient = loss.calculate_gradient(labels[key], feat_list, weights)

                # plug into gradient update and update weights
                sgd.apply_gradient_update(gradient, 1)

    for i in range(len(indexer)):
        print(indexer.get_object(i), sgd.get_final_weights()[i])

    pred = PersonClassifier(sgd.get_final_weights(), indexer)
    return pred

def sigmoid(z):
    """Implement logistic regression here. Takes two numpy arrays, calculates their dot product,
    and plugs it into sigmoid formula"""
    # z = np.dot(weights, inputs)
    out = np.exp(z)/(1+np.exp(z))
    return out

# def calculate_gradient(sig_val, label, feat_vect, indexer):
#     # feat_vect does not hold any zero values, but instead it is used to index
#     # which features are applicable to a particular word. ie feat_vec == [4,5] means
#     # the 4th and 5th features are 1, and all others are 0
#     grad = np.zeros(len(indexer))
#
#     # The cost function being used is (sigma(wx) - y) where sigma is the logistic value, and y is the correct label
#     # This function keeps all values zero except those that have features show up
#     for index_val in feat_vect:
#         grad[index_val] = (sig_val - label)

def evaluate_classifier(exs, classifier):
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    for ex in exs:
        for idx in range(0, len(ex)):
            prediction = classifier.predict(ex.tokens, idx)
            if prediction == ex.labels[idx]:
                num_correct += 1
            if prediction == 1:
                num_pred += 1
            if ex.labels[idx] == 1:
                num_gold += 1
            if prediction == 1 and ex.labels[idx] == 1:
                num_pos_correct += 1
            num_total += 1
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
    prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
    rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec/(prec + rec) if prec > 0 and rec > 0 else 0.0
    print("Precision: %i / %i = %f" % (num_pos_correct, num_pred, prec))
    print("Recall: %i / %i = %f" % (num_pos_correct, num_gold, rec))
    print("F1: %f" % f1)


# Runs prediction on exs and writes the outputs to outfile, one token per line
def predict_write_output_to_file(exs, classifier, outfile):
    f = open(outfile, 'w')
    for ex in exs:
        for idx in range(0, len(ex)):
            prediction = classifier.predict(ex.tokens, idx)
            f.write(ex.tokens[idx] + " " + repr(int(prediction)) + "\n")
        f.write("\n")
    f.close()


def main():
    start_time = time.time()  # saves start time for calculation of running time
    args = _parse_args()  # _parse_args() uses argparse to determine constraints (such as model to run)
    print(args)

    # Load the training and test data
    train_class_exs = list(transform_for_classification(read_data(args.train_path)))
    dev_class_exs = list(transform_for_classification(read_data(args.dev_path)))

    # Train the model
    # if args.model == "BAD":
    #     classifier = train_count_based_binary_classifier(train_class_exs)
    # else:
    #     classifier = train_classifier(train_class_exs)
    classifier = train_classifier(train_class_exs)

    print("Data reading and training took %f seconds" % (time.time() - start_time))
    # Evaluate on training, development, and test data
    print("===Train accuracy===")
    evaluate_classifier(train_class_exs, classifier)
    print("===Dev accuracy===")
    evaluate_classifier(dev_class_exs, classifier)
    if args.run_on_test:
        print("Running on test")
        test_exs = list(transform_for_classification(read_data(args.blind_test_path)))
        predict_write_output_to_file(test_exs, classifier, args.test_output_path)
        print("Wrote predictions on %i labeled sentences to %s" % (len(test_exs), args.test_output_path))


if __name__ == '__main__':
    main()

