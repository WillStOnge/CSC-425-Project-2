import math, numpy as np
from spam_filter import (
    generate_class_log_prior,
    Classification,
    NUM_CLASSES,
    MOST_COMMON_WORD,
)

class GaussianNB:
    mean = np.zeros((NUM_CLASSES, MOST_COMMON_WORD))
    stddev = np.zeros((NUM_CLASSES, MOST_COMMON_WORD))
    class_log_prior = np.zeros(NUM_CLASSES)


    def __init__(self, features, labels):
        """ Trains the model using Gaussian NB. """

        # Calculate the means
        for i in range(MOST_COMMON_WORD):
            sum_ham = 0
            sum_spam = 0
            for j in range(len(features)):
                if labels[j] == 0:
                    sum_ham += features[j][i]
                if labels[j] == 1:
                    sum_spam += features[j][i]

            self.mean[Classification.HAM.value][i] = sum_ham / (len(labels) - np.count_nonzero(labels))
            self.mean[Classification.SPAM.value][i] = sum_spam / np.count_nonzero(labels)

        # Calculate the standard deviations
        for i in range(MOST_COMMON_WORD):
            seq_ham = 0
            seq_spam = 0

            for j in range(features.shape[0]):
                if labels[j] == 0:
                    seq_ham += math.pow((float(features[j][i]) - self.mean[Classification.HAM.value][i]), 2)
                if labels[j] == 1:
                    seq_spam += math.pow((float(features[j][i]) - self.mean[Classification.SPAM.value][i]), 2)

            self.stddev[Classification.HAM.value][i] = math.sqrt(seq_ham / (len(labels) - np.count_nonzero(labels)))
            self.stddev[Classification.SPAM.value][i] = math.sqrt(seq_spam / np.count_nonzero(labels))
        
        # Calculate class_log_prior
        self.class_log_prior = generate_class_log_prior(labels)


    def predict(self, features):
        """ Classify an array of emails feature vectors. """
        classes = np.zeros(features.shape[0])

        # Loop through each email
        for i in range(features.shape[0]):
            ham_prob = 0
            spam_prob = 0
            feature_log_prob = np.zeros((NUM_CLASSES, MOST_COMMON_WORD))

            # Calculate feature_log_prob
            for j in range(MOST_COMMON_WORD):
                if self.stddev[Classification.HAM.value][j] != 0 and self.mean[Classification.HAM.value][j] != 0:
                    var = float(self.stddev[Classification.HAM.value][j]) ** 2
                    denom = (2 * math.pi * var) ** 0.5
                    num = math.exp(-(float(features[i][j]) - float(self.mean[Classification.HAM.value][j])) ** 2 / (2 * var))
                    feature_log_prob[Classification.HAM.value][j] = num / denom
                if self.stddev[Classification.SPAM.value][j] != 0 and self.mean[Classification.SPAM.value][j] != 0:
                    var = float(self.stddev[Classification.SPAM.value][j]) ** 2
                    denom = (2 * math.pi * var) ** .5
                    num = math.exp(-(float(features[i][j]) - float(self.mean[Classification.SPAM.value][j])) ** 2 / (2 * var))
                    feature_log_prob[Classification.SPAM.value][j] = num / denom
            
            # Calculate probabilities
            for j in range(len(feature_log_prob[Classification.HAM.value])):
                if feature_log_prob[Classification.HAM.value][j] != 0:
                    ham_prob += math.log(feature_log_prob[Classification.HAM.value][j]) + math.log(abs(self.class_log_prior[Classification.HAM.value]))
                if feature_log_prob[Classification.SPAM.value][j] != 0:
                    spam_prob += math.log(feature_log_prob[Classification.SPAM.value][j]) + math.log(abs(self.class_log_prior[Classification.SPAM.value]))

            # Determine which probability is higher.
            classes[i] = (
                Classification.HAM.value
                if ham_prob > spam_prob
                else Classification.SPAM.value
            )
        return classes
