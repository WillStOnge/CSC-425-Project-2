import math, enum, numpy as np
from spam_filter import (
    generate_class_log_prior,
    Classification,
    NUM_CLASSES,
    MOST_COMMON_WORD,
    SMOOTH_ALPHA,
)

class MultinomialNB:
    """ Multinomial Naive Bayes """
    class_log_prior: np.ndarray
    feature_log_prob: np.ndarray

    def __init__(self, features: np.array, labels: np.array):
        """ Trains the model using Multinomial NB. """
        self.feature_log_prob = np.zeros((NUM_CLASSES, MOST_COMMON_WORD))
        # Get distribution of ham vs spam in training set.
        self.class_log_prior = generate_class_log_prior(labels)

        # Numerator of conditional probabilty
        ham_words = np.tile(SMOOTH_ALPHA, MOST_COMMON_WORD)
        spam_words = np.tile(SMOOTH_ALPHA, MOST_COMMON_WORD)

        # Denominator of conditional probabilty
        ham_sum = MOST_COMMON_WORD * SMOOTH_ALPHA
        spam_sum = MOST_COMMON_WORD * SMOOTH_ALPHA

        for i in range(features.shape[0]):
            if labels[i] == Classification.HAM.value:
                ham_words += features[i]
                ham_sum += np.sum(features[i])
            else:
                spam_words += features[i]
                spam_sum += np.sum(features[i])

        self.feature_log_prob[Classification.HAM.value] = np.log(ham_words / ham_sum)
        self.feature_log_prob[Classification.SPAM.value] = np.log(spam_words / spam_sum)


    def predict(self, features: np.array):
        """ Classify an array of emails feature vectors. """
        classes = np.zeros(features.shape[0])

        for i in range(features.shape[0]):
            # Calculate the probability that the email is spam or ham.
            ham_prob = np.sum(self.feature_log_prob[Classification.HAM.value] * features[i]) + self.class_log_prior[Classification.HAM.value]
            spam_prob = np.sum(self.feature_log_prob[Classification.SPAM.value] * features[i]) + self.class_log_prior[Classification.SPAM.value]

            classes[i] = (
                Classification.HAM.value
                if ham_prob > spam_prob
                else Classification.SPAM.value
            )

        return classes

