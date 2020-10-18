import os
import math
import numpy as np

most_common_word = 3000
# avoid 0 terms in features
smooth_alpha = 1.0
class_num = 2  # we have only two classes: ham and spam
class_log_prior = [0.0, 0.0]  # probability for two classes
feature_log_prob = np.zeros(
    (class_num, most_common_word)
)  # feature parameterized probability
SPAM = 1  # spam class label
HAM = 0  # ham class label

# mean and standard deviation for Gaussian distribution
mean = np.zeros((class_num, most_common_word))
std = np.zeros((class_num, most_common_word))


class Gauss:
    # Gaussian Naive Bayes
    def GaussianNB(self, features, labels):
        # calculate the means
        for i in range(most_common_word):
            sum_ham = 0
            sum_spam = 0
            for j in range(len(features)):
                sum_ham += features[j][i]
                sum_spam += features[j][i]
            mean[0][i] = sum_ham / (len(labels) - np.count_nonzero(labels))
            mean[1][i] = sum_spam / np.count_nonzero(labels)
        # calculate the standard deviations
        for x in range(most_common_word):
            seq_ham = 0
            seq_spam = 0
            for y in range(len(features)):
                seq_ham += math.pow((features[y][x] - mean[0][x]), 2)
                seq_spam += math.pow((features[y][x] - mean[1][x]), 2)
            std[0][x] = math.sqrt(seq_ham / (len(labels) - np.count_nonzero(labels)))
            std[1][x] = math.sqrt(seq_ham / np.count_nonzero(labels))

    # Gaussian Naive Bayes prediction
    def GaussianNB_predict(self, features):
        classes = np.zeros(len(features))

        ham_prob = 0.0
        spam_prob = 0.0
        """//calculate the Gaussian value for each feature
             and summ over one specific file
		/**
		 * nested loop over features with i and j
		 * calculate ham_prob and spam_prob
		 * 1.0/(std*Math.sqrt(2.0*Math.PI))*
		   Math.exp(-(Math.pow((features[i][j]-mean), 2)/2.0*Math.pow(std, 2)));
		 * if ham_prob > spam_prob
		 * HAM
		 * else SPAM
		 * return  classes
		 */"""
        for i in range(most_common_word):
            for j in range(len(features)):

        return classes
