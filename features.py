from utils import maybe_add_feature, Indexer, score_indexed_features
import re
from collections import defaultdict
import numpy as np

class Features():
    def __init__(self):
        self.feature_list = \
            ["isCap",    # is a Capitalized word
            "isNotCap",     # Negative feature for non capitalized words
            "isPoss",   # is possessive (next token is 's)
            "isFirstWord",   # is the first word in the sentence
            "hasPronoun",     # is a pronoun?? or sentence has a pronoun
            "manySyl",    # Has greater than x syllables
            "isInitials",   #Is representing initials: R.W.
            "isIs",               # the word = "is"
            "isArticle",          # The word is an article (the, a, an)
            "beginsWithNum",      # The word begins with numbers
            "hasNumbers",         # The token contains numbers
            "1Syl",               # Token has 1 syllable
            "2Syl",
            "3Syl",
            "4PlusSyl",           # Token has 4+ syllables
            ]


def init_features(indexer):
    features = Features()
    featlist = features.feature_list

    feats = []
    for feature in featlist:
        # use the maybe_add_feature() function to initialize all of the features into the indexer
        feats = maybe_add_feature(feats, indexer, True, feature)
    return indexer

def get_applicable_feats(sentence):
    """
    Creates a dictionary with each token of sentence as a key, and the value for each key
    is the name of every feature applicable to that word.
    :param sentence:
    :return:
    """

    articles = ["the", "that", "this", "an", "a"]

    feats_per_word = defaultdict(list)

    for i, token in enumerate(sentence):
        # initialize the token in the dictionary with value being empty list
        # if not feats_per_word[token]:
        #     feats_per_word[token] = []

        # NOTE: FIRST WORD OF LIST IS THE TOKEN
        feats_per_word[i].append(token)
        # Check if token is first word of sentence
        if i == 0:
            feats_per_word[i].append("isFirstWord")

        #Check capitalization:
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
        if i+1 < len(sentence):
            # If following token is "'s", then the current token is possessive
            if sentence[i+1].lower() == "'s":
                feats_per_word[i].append("isPoss")

        if check_initials(token):
            feats_per_word[i].append("isInitials")

        syls = sylco(token)

        if syls < 2:
            feats_per_word[i].append("1Syl")
        elif syls == 2:
            feats_per_word[i].append("2Syl")
        elif syls == 3:
            feats_per_word[i].append("3Syl")
        else:
            feats_per_word[i].append("4PlusSyl")

    return feats_per_word

def starts_with_num(token):
    numpattern = re.compile(r"[0-9]+[a-zA-Z]+")
    if re.search(numpattern, token):
        return True
    else: return False

def contains_num(token):
    numpattern = re.compile(r"[0-9]+")
    if re.search(numpattern, token):
        return True
    else: return False

def check_initials(token):
    initials = re.compile(r"[A-Z][.]")
    if re.search(initials, token):
        return True
    else: return False

def get_score_from_feats(feature_dict, indexer, weights):
    """

    :param feature_dict: Dictionary with keys representing position in sentence, values are lists with first element being the token
    :return: score for each word?
    """
    scores = []
    words = []

    for position, featureset in feature_dict.items():
        feats = []  # Reset feats list for each new word

        # if len(word) is 1, that means it doesn't have any attached features. Don't need to run it through maybe_add_feature
        if len(featureset) > 1:
            for i, feature in enumerate(featureset):
                feats = maybe_add_feature(feats, indexer, False, feature)

            scores.append(score_indexed_features(feats, weights))
            words.append(featureset[0])

    return zip(scores,words)


# Todo: Check with prof durrett to see if I can use this
def sylco(word):
    # Taken from here: https://eayd.in/?p=232
    word = word.lower()
    # exception_add are words that need extra syllables
    # exception_del are words that need less syllables

    exception_add = ['serious','crucial']
    exception_del = ['fortunately','unfortunately']

    co_one = ['cool','coach','coat','coal','count','coin','coarse','coup','coif','cook','coign','coiffe','coof','court']
    co_two = ['coapt','coed','coinci']

    pre_one = ['preach']

    syls = 0 #added syllable number
    disc = 0 #discarded syllable number

    #1) if letters < 3 : return 1
    if len(word) <= 3 :
        syls = 1
        return syls

    #2) if doesn't end with "ted" or "tes" or "ses" or "ied" or "ies", discard "es" and "ed" at the end.
    # if it has only 1 vowel or 1 set of consecutive vowels, discard. (like "speed", "fled" etc.)

    if word[-2:] == "es" or word[-2:] == "ed" :
        doubleAndtripple_1 = len(re.findall(r'[eaoui][eaoui]',word))
        if doubleAndtripple_1 > 1 or len(re.findall(r'[eaoui][^eaoui]',word)) > 1 :
            if word[-3:] == "ted" or word[-3:] == "tes" or word[-3:] == "ses" or word[-3:] == "ied" or word[-3:] == "ies" :
                pass
            else :
                disc+=1

    #3) discard trailing "e", except where ending is "le"

    le_except = ['whole','mobile','pole','male','female','hale','pale','tale','sale','aisle','whale','while']

    if word[-1:] == "e" :
        if word[-2:] == "le" and word not in le_except :
            pass

        else :
            disc+=1

    #4) check if consecutive vowels exists, triplets or pairs, count them as one.

    doubleAndtripple = len(re.findall(r'[eaoui][eaoui]',word))
    tripple = len(re.findall(r'[eaoui][eaoui][eaoui]',word))
    disc+=doubleAndtripple + tripple

    #5) count remaining vowels in word.
    numVowels = len(re.findall(r'[eaoui]',word))

    #6) add one if starts with "mc"
    if word[:2] == "mc" :
        syls+=1

    #7) add one if ends with "y" but is not surrouned by vowel
    if word[-1:] == "y" and word[-2] not in "aeoui" :
        syls +=1

    #8) add one if "y" is surrounded by non-vowels and is not in the last word.

    for i,j in enumerate(word) :
        if j == "y" :
            if (i != 0) and (i != len(word)-1) :
                if word[i-1] not in "aeoui" and word[i+1] not in "aeoui" :
                    syls+=1

    #9) if starts with "tri-" or "bi-" and is followed by a vowel, add one.

    if word[:3] == "tri" and word[3] in "aeoui" :
        syls+=1

    if word[:2] == "bi" and word[2] in "aeoui" :
        syls+=1

    #10) if ends with "-ian", should be counted as two syllables, except for "-tian" and "-cian"

    if word[-3:] == "ian" :
    #and (word[-4:] != "cian" or word[-4:] != "tian") :
        if word[-4:] == "cian" or word[-4:] == "tian" :
            pass
        else :
            syls+=1

    #11) if starts with "co-" and is followed by a vowel, check if exists in the double syllable dictionary, if not, check if in single dictionary and act accordingly.

    if word[:2] == "co" and word[2] in 'eaoui' :

        if word[:4] in co_two or word[:5] in co_two or word[:6] in co_two :
            syls+=1
        elif word[:4] in co_one or word[:5] in co_one or word[:6] in co_one :
            pass
        else :
            syls+=1

    #12) if starts with "pre-" and is followed by a vowel, check if exists in the double syllable dictionary, if not, check if in single dictionary and act accordingly.

    if word[:3] == "pre" and word[3] in 'eaoui' :
        if word[:6] in pre_one :
            pass
        else :
            syls+=1

    #13) check for "-n't" and cross match with dictionary to add syllable.

    negative = ["doesn't", "isn't", "shouldn't", "couldn't","wouldn't"]

    if word[-3:] == "n't" :
        if word in negative :
            syls+=1
        else :
            pass

    #14) Handling the exceptional words.

    if word in exception_del :
        disc+=1

    if word in exception_add :
        syls+=1

    # calculate the output
    return numVowels - disc + syls

if __name__ == '__main__':

    # to consider: Do I want negative score values for words that definitely aren't names?
    # Todo: Try making "not capitalized" a feature. This should set the weights heavily negative for this feature
    indexer = Indexer()

    indexer = init_features(indexer)
    weights = np.random.rand(len(indexer))

    sentence = "Bill 's four big dogs were named Doug Susan Bill J.R. and Andre3000".split(" ")
    feature_dict = get_applicable_feats(sentence)
    get_score_from_feats(feature_dict, indexer, weights)
