import math, enum, numpy as np


class Classification(enum.Enum):
    HAM = 0
    SPAM = 1


NUM_CLASSES = len(Classification)
MOST_COMMON_WORD = 3000
SMOOTH_ALPHA = 1.0


class MultinomialNaiveBayes:
    """ Multinomial Naive Bayes """

    class_log_prior: np.ndarray
    feature_log_prob: np.ndarray

    def __init__(self, features: np.array, labels: np.array):
        """ Trains the model using Multinomial NB. """
        self.class_log_prior = np.zeros(NUM_CLASSES, dtype=np.float64)
        self.feature_log_prob = np.zeros((NUM_CLASSES, MOST_COMMON_WORD))

        # Get distribution of ham vs spam in training set.
        self.class_log_prior[Classification.HAM.value] = math.log(
            np.sum(labels == Classification.HAM.value)
        )
        self.class_log_prior[Classification.SPAM.value] = math.log(
            np.sum(labels == Classification.SPAM.value)
        )

        # Numerator of conditional probabilty
        ham_words = np.tile(SMOOTH_ALPHA, MOST_COMMON_WORD)
        spam_words = np.tile(SMOOTH_ALPHA, MOST_COMMON_WORD)

        # Denominator of conditional probabilty
        ham_sum = MOST_COMMON_WORD * SMOOTH_ALPHA
        spam_sum = MOST_COMMON_WORD * SMOOTH_ALPHA

        for i in range(features.shape[0]):
            for j in range(features.shape[1]):
                if labels[i] == Classification.HAM.value:
                    ham_words[j] += features[i][j]
                    ham_sum += features[i][j]
                else:
                    spam_words[j] += features[i][j]
                    spam_sum += features[i][j]

        for i in range(features.shape[1]):
            self.feature_log_prob[Classification.HAM.value][i] = math.log(
                ham_words[i] / ham_sum
            )
            self.feature_log_prob[Classification.SPAM.value][i] = math.log(
                spam_words[i] / spam_sum
            )

    def predict(self, features: np.array):
        """ Classify an array of emails feature vectors. """
        classes = np.zeros(features.shape[0])

        word_occurences = np.sum(features, axis=0)

        # Loop through each email
        for i in range(features.shape[0]):
            ham_prob = 0
            spam_prob = 0
            # Calculate the probability that the email is spam or ham.
            for j in range(features.shape[1]):
                ham_prob += (
                    self.feature_log_prob[Classification.HAM.value][j] * features[i][j]
                )
                spam_prob += (
                    self.feature_log_prob[Classification.SPAM.value][j] * features[i][j]
                )

            # Add class_log_prior value
            ham_prob += self.class_log_prior[Classification.HAM.value]
            spam_prob += self.class_log_prior[Classification.SPAM.value]

            # Determine which probability is higher.
            classes[i] = (
                Classification.HAM.value
                if ham_prob > spam_prob
                else Classification.SPAM.value
            )

        return classes

