import math
import numpy as np
import ipdb
from sklearn.naive_bayes import BernoulliNB as sbnb
from spam_filter import (
    generate_class_log_prior,
    Classification,
    SMOOTH_ALPHA,
    MOST_COMMON_WORD,
    NUM_CLASSES,
)


class BernoulliNB:

    feature_log_prob: np.ndarray
    # Bernoulli Naive Bayes
    def __init__(self, features, labels):

        # convert features to l0-norm
        self.features = features
        self.features[features != 0] = 1

        self.feature_log_prob = np.zeros((NUM_CLASSES, MOST_COMMON_WORD))

        # Get distribution of ham vs spam in training set.
        self.class_log_prior = generate_class_log_prior(labels)

        # Numerator of conditional probabilty
        ham_words = np.zeros(MOST_COMMON_WORD)
        spam_words = np.zeros(MOST_COMMON_WORD)

        # Denominator of conditional probabilty
        ham_sum = 0
        spam_sum = 0

        for i in range(features.shape[0]):
            for j in range(features.shape[1]):
                if labels[i] == Classification.HAM.value:
                    ham_words[j] += features[i][j] + SMOOTH_ALPHA
                else:
                    spam_words[j] += features[i][j] + SMOOTH_ALPHA

        ham_sum = ham_words.sum() + SMOOTH_ALPHA * 2
        spam_sum = spam_words.sum() + SMOOTH_ALPHA * 2

        for i in range(features.shape[1]):
            self.feature_log_prob[Classification.HAM.value][i] = ham_words[i] / ham_sum

            self.feature_log_prob[Classification.SPAM.value][i] = (
                spam_words[i] / spam_sum
            )

    def predict(self, features):
        """ Classify an array of emails feature vectors. """
        classes = np.zeros(features.shape[0])
        ham_prob = 0
        spam_prob = 0
        for i in range(features.shape[0]):
            for j in range(features.shape[1]):
                ham_prob += math.log(
                    self.feature_log_prob[Classification.HAM.value][j]
                ) * features[i][j] + abs(1 - features[i][j]) * math.log(
                    1 - self.feature_log_prob[Classification.HAM.value][j]
                )

                spam_prob += math.log(
                    self.feature_log_prob[Classification.SPAM.value][j]
                ) * features[i][j] + abs(1 - features[i][j]) * math.log(
                    1 - self.feature_log_prob[Classification.SPAM.value][j]
                )

            classes[i] = (
                Classification.HAM.value
                if ham_prob > spam_prob
                else Classification.SPAM.value
            )
        return classes
