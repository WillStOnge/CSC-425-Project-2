import math
import numpy as np
import ipdb
from sklearn.naive_bayes import BernoulliNB as sbnb
from sklearn.utils.extmath import safe_sparse_dot
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

        # Fix the labels, so you can actually do the math

        # Count of each feature, same as co
        self.class_log_prior = generate_class_log_prior(labels)

        self.class_count = (
            np.array([(labels == 0).sum(), (labels == 1).sum()]) + SMOOTH_ALPHA * 2
        )

        # Translate 1D array into 2D array with first being the inverse of the second array.
        # Because a lot of numpy magic needs the arrays to be equal.
        fixed_labels = np.array([1 - labels, labels])

        self.feature_count = np.zeros((NUM_CLASSES, MOST_COMMON_WORD), dtype=np.float64)
        self.feature_count += np.dot(fixed_labels, self.features) + SMOOTH_ALPHA

        self.feature_log_prob = np.log(self.feature_count) - np.log(
            self.class_count.reshape(-1, 1)
        )

        ipdb.set_trace()

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
