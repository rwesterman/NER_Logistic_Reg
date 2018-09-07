from utils import maybe_add_feature, Indexer, score_indexed_features
import re
from collections import defaultdict
import numpy as np


class Features():
    def __init__(self):
        self.feature_list = \
            ["isCap",  # is a Capitalized word
             "isNotCap",  # Negative feature for non capitalized words
             "isPoss",  # is possessive (next token is 's)
             "isFirstWord",  # is the first word in the sentence
             "hasPronoun",  # is a pronoun?? or sentence has a pronoun
             # "manySyl",  # Has greater than x syllables
             "isInitials",  # Is representing initials: R.W.
             "isIs",  # the word = "is"
             "isArticle",  # The word is an article (the, a, an)
             "beginsWithNum",  # The word begins with numbers
             "hasNumbers",  # The token contains numbers
             # "1Syl",  # Token has 1 syllable
             # "2Syl",
             # "3Syl",
             # "4PlusSyl",  # Token has 4+ syllables
             "hasTitle",     # token is preceded by a title
             ]

def add_currword_to_indexer(token, indexer):
    add_or_not = not(indexer.contains(token))
    maybe_add_feature([], indexer, True, token)
    return indexer


def init_features(indexer):
    features = Features()
    featlist = features.feature_list

    feats = []
    for feature in featlist:
        # use the maybe_add_feature() function to initialize all of the features into the indexer
        feats = maybe_add_feature(feats, indexer, True, feature)
    return indexer

def get_applicable_feats(sentence, stop_words, indexer):
    """
    Creates a dictionary with each token of sentence as a key, and the value for each key
    is the name of every feature applicable to that word.
    :param indexer:
    :param sentence:
    :return:
    """

    # List of stop words is read in from file and passed to this function
    articles = stop_words
    titles = ["mr.", "mrs.", "ms.", "miss", "president", "diplomat", "ambassador", "bishop", "father",
              "minister", "dr.", "dr", "doctor", "mister"]

    feats_per_word = defaultdict(list)

    for i, token in enumerate(sentence):

        # Add the token itself to the indexer if it's not already present.
        # This will help track how often a particular word shows up.
        indexer = add_currword_to_indexer(token, indexer)

        # NOTE: FIRST WORD OF LIST IS THE TOKEN
        feats_per_word[i].append(token)
        # Check if token is first word of sentence
        if i == 0:
            feats_per_word[i].append("isFirstWord")

        # Check capitalization:
        if token.istitle():
            feats_per_word[i].append("isCap")
        else:
            feats_per_word[i].append("isNotCap")

        # Check if the word is an article
        if token.lower() in articles:
            feats_per_word[i].append("isArticle")

        # Check if token starts with numbers, or else if it has numbers at all
        if starts_with_num(token):
            # if the token begins with a number, then it also "hasNumbers", so add both
            feats_per_word[i].append("hasNumbers")
            feats_per_word[i].append("beginsWithNum")
        elif contains_num(token):
            # If the token doesn't begin with a number, it might still contain a number, so check that independently
            feats_per_word[i].append("hasNumbers")

        # check if the next token is "'s", if so mark current token as possessive
        # Make sure current token isn't the last token in the sentence before testing
        if i + 1 < len(sentence):
            # If following token is "'s", then the current token is possessive
            if sentence[i + 1].lower() == "'s":
                feats_per_word[i].append("isPoss")

        if i - 1 >= 0:
            if sentence[i-1].lower() in titles:
                feats_per_word[i].append("hasTitle")

        if check_initials(token):
            feats_per_word[i].append("isInitials")

    return feats_per_word, indexer


def starts_with_num(token):
    numpattern = re.compile(r"[0-9]+[a-zA-Z]+")
    if re.search(numpattern, token):
        return True
    else:
        return False


def contains_num(token):
    numpattern = re.compile(r"[0-9]+")
    if re.search(numpattern, token):
        return True
    else:
        return False


def check_initials(token):
    initials = re.compile(r"[A-Z][.]")
    if re.search(initials, token):
        return True
    else:
        return False

