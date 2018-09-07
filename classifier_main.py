# classifier_main.py

import argparse
import sys
import time
from nerdata import *
from utils import *
from optimizers import *

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

    # Makes a prediction for token at position idx in the given PersonExample
    def predict(self, tokens, idx):
        raise Exception("Implement me!")


def train_classifier(ner_exs):
    # Todo: Implement a training method here, follow steps below:
    for ex in ner_exs:
        create_features(ex)


    # use counter to keep track of gradient (?)
    # can do vector implementation, using dot product instead of looping over each element (?)
    # features to use: word indicators, capitalization, possessives, any other word features (?)

    # Probably want to implement a sigmoid (logistic regression) classifier here.
    # raise Exception("Implement me!")
    # Will need to calculate the gradient as well.
    pass

def sigmoid(weights, inputs):
    """Implement logistic regression here. Takes two numpy arrays, calculates their dot product,
    and plugs it into sigmoid formula"""
    z = np.dot(weights, inputs)
    out = np.exp(z)/(1+np.exp(z))
    print(out)

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
    if args.model == "BAD":
        classifier = train_count_based_binary_classifier(train_class_exs)
    else:
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
    # main()
    index = Indexer()
    print(index)
    maybe_add_feature([0,1,5], index, True, "Capitalized")
    print(index)
    maybe_add_feature([8,10], index, True, "Possessive")
    print(index.index_of("Capitalized"))
    print(index.index_of("Possessive"))
    print(index.get_object(1))

