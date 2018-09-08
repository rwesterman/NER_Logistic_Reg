from utils import maybe_add_feature, Indexer, score_indexed_features
import re
from collections import defaultdict
import numpy as np

# Todo: ADD BIAS TERM!!!

class Features():
    def __init__(self):
        self.feature_list = [
             "isCap",  # is a Capitalized word
             "isNotCap",  # Negative feature for non capitalized words
             "allCaps",     # allCaps
             "isPoss",  # is possessive (next token is 's)
             "isFirstWord",  # is the first word in the sentence
             # "hasPronoun",  # is a pronoun?? or sentence has a pronoun
             "isInitials",  # Is representing initials: R.W.
             "isArticle",  # NOT WORKING
             "beginsWithNum",  # The word begins with numbers
             "hasNumbers",  # The token contains numbers
             # "endS",        #
             "hasTitle",     # token is preceded by a title
             "onlyLetters",  # checks that token only has letters
             "hasBias",     # general bias term across all words. This should be strongly negative
             "isTitle",      # if current word is title
             "noAlphaNum",
             "hasApostrophe",
             "hasHyphen",
             ]

def add_currword_to_indexer(token, indexer, train_flag):
    add_or_not = not(indexer.contains(token)) and train_flag
    maybe_add_feature([], indexer, add_or_not, token)
    return indexer

def init_features(indexer):
    features = Features()
    featlist = features.feature_list

    feats = []
    for feature in featlist:
        # use the maybe_add_feature() function to initialize all of the features into the indexer
        feats = maybe_add_feature(feats, indexer, True, feature)
    return indexer

def get_applicable_feats(sentence, stop_words, indexer, curr_feats, train_flag):
    """
    Creates a dictionary with each token of sentence as a key, and the value for each key
    is the name of every feature applicable to that word.
    :param indexer:
    :param sentence:
    :param indexer: Indexer() object used to add new features to indexer
    :param curr_feats: The current state of the "List of Lists" of features for every word
    :param train_flag: This flag indicates if the featurization is occuring during training or prediction
    :return:
    """

    # List of stop words is read in from file and passed to this function
    articles = stop_words
    titles = ["mr.", "mrs.", "ms.", "miss", "president", "diplomat", "ambassador", "bishop", "father",
              "minister", "dr.", "dr", "doctor", "mister", "lord", "duke", "king", "queen",
              "senator"]

    for i, token in enumerate(sentence):
        # empty list feats_per_word will store the features for a single token
        feats_per_word = []
        # Add the token itself to the indexer if it's not already present and this is a training run
        # This will help track how often a particular word shows up.
        add_or_not = not (indexer.contains(token)) and train_flag
        maybe_add_feature([], indexer, add_or_not, token)

        # Check the bounds
        # Trying Bigrams here as well
        if i > 0:
            # Do similar for context words immediately surrounding it. ("prevWord", "nextWord")
            add_or_not = not(indexer.contains(sentence[i-1])) and train_flag
            prev_word = "prev{}".format(sentence[i-1].lower())
            maybe_add_feature([], indexer, add_or_not, prev_word)




        if i + 1 < len(sentence):
            add_or_not = not (indexer.contains(sentence[i + 1])) and train_flag
            next_word = "next{}".format(sentence[i + 1].lower())
            maybe_add_feature([], indexer, add_or_not, next_word)



        # NOTE: FIRST WORD OF LIST IS THE TOKEN
        feats_per_word.append(token)

        # next element is bias
        feats_per_word.append("hasBias")

        # Check if token is first word of sentence
        if i == 0:
            feats_per_word.append("isFirstWord")

        # Check capitalization:
        if token.istitle():
            feats_per_word.append("isCap")
        elif token.isupper():
            feats_per_word.append("allCaps")
        else:
            feats_per_word.append("isNotCap")

        # Check if the word is an article
        if token.lower() in articles:
            feats_per_word.append("isArticle")

        # Check if token starts with numbers, or else if it has numbers at all
        if starts_with_num(token):
            # if the token begins with a number, then it also "hasNumbers", so add both
            feats_per_word.append("hasNumbers")
            feats_per_word.append("beginsWithNum")
        elif contains_num(token):
            # If the token doesn't begin with a number, it might still contain a number, so check that independently
            feats_per_word.append("hasNumbers")

        # check if the next token is "'s", if so mark current token as possessive
        # Make sure current token isn't the last token in the sentence before testing
        if i + 1 < len(sentence):
            # If following token is "'s", then the current token is possessive
            if sentence[i + 1].lower() == "'s":
                feats_per_word.append("isPoss")

        if i > 0:
            if sentence[i-1].lower() in titles:
                feats_per_word.append("hasTitle")

        if token.lower() in titles:
            feats_per_word.append("isTitle")

        if check_initials(token):
            feats_per_word.append("isInitials")

        if token.lower()[-1] == "s":
            feats_per_word.append("endS")

        if only_letters(token):
            feats_per_word.append("onlyLetters")

        if no_word_match(token):
            feats_per_word.append("noAlphaNum")

        if "-" in token:
            feats_per_word.append("hasHyphen")

        if "'" in token:
            feats_per_word.append("hasApostrophe")

        curr_feats.append(feats_per_word)

    # return the appended curr_feats list
    return curr_feats


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

def only_letters(token):
    letters = re.compile(r"^[a-zA-Z]$")
    if re.search(letters, token):
        return True
    else:
        return False

def no_word_match(token):
    """Only matches tokens that do not match alphanumeric characters"""
    search = re.compile(r"^\W+$")
    if re.search(search, token):
        return True
    else: return False