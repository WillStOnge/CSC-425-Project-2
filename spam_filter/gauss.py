import math

import numpy as np

most_common_word = 3000
# avoid 0 terms in features
smooth_alpha = 1.0
class_num = 2  # we have only two classes: ham and spam
class_log_prior = [0.0, 0.0]  # probability for two classes
#feature_log_prob = np.zeros((class_num, most_common_word))  # feature parameterized probability
SPAM = 1  # spam class label
HAM = 0  # ham class label

# mean and standard deviation for Gaussian distribution
mean = np.zeros((class_num, most_common_word))
std = np.zeros((class_num, most_common_word))


class Gauss:
    # Gaussian Naive Bayes
    def GaussianNBMe(self, features, labels):
        # calculate the means
        for i in range(most_common_word):
            sum_ham = 0
            sum_spam = 0
            for j in range(len(features)):
                if labels[j] == 0:
                    sum_ham += features[j][i]
                if labels[j] == 1:
                    sum_spam += features[j][i]
            mean[0][i] = sum_ham / (len(labels) - np.count_nonzero(labels))
            mean[1][i] = sum_spam / np.count_nonzero(labels)
        # calculate the standard deviations
        for x in range(most_common_word):
            seq_ham = 0
            seq_spam = 0
            for y in range(len(features)):
                if labels[y] == 0:
                    seq_ham += math.pow((float(features[y][x]) - mean[0][x]), 2)
                if labels[y] == 1:
                    seq_spam += math.pow((float(features[y][x]) - mean[1][x]), 2)
            std[0][x] = math.sqrt(seq_ham / (len(labels) - np.count_nonzero(labels)))
            std[1][x] = math.sqrt(seq_spam / np.count_nonzero(labels))
        # calculate the priors
        hamCount = float(np.size(labels) - np.count_nonzero(labels)) / float(np.size(labels))
        spamCount = float(np.count_nonzero(labels)) / float(np.size(labels))
        class_log_prior[0] = hamCount
        class_log_prior[1] = spamCount

    # Gaussian Naive Bayes prediction
    def GaussianNB_predict(self, features):
        classes = np.zeros(len(features))
        for i in range(len(features)):
            ham_prob = 0
            spam_prob = 0
            feature_log_prob = np.zeros((class_num, most_common_word))
            for j in range(most_common_word):
                if std[0][j] != 0 and mean[0][j] != 0:
                    var = float(std[0][j])**2
                    denom = (2*math.pi*var)**.5
                    num = math.exp(-(float(features[i][j]) - float(mean[0][j])) ** 2 / (2 * var))
                    feature_log_prob[0][j] = num/denom
                    #feature_log_prob[0][j] = (1.0 / (std[0][j] * math.sqrt(2.0 * math.pi))) * (math.exp(-(math.pow((features[i][j] - mean[0][j]), 2) / 2.0 * math.pow(std[0][j], 2))))
                if std[1][j] != 0 and mean[1][j] != 0:
                    var = float(std[1][j]) ** 2
                    denom = (2 * math.pi * var) ** .5
                    num = math.exp(-(float(features[i][j]) - float(mean[1][j])) ** 2 / (2 * var))
                    feature_log_prob[1][j] = num/denom
                    #feature_log_prob[1][j] = (1.0 / (std[1][j] * math.sqrt(2.0 * math.pi))) * (math.exp(-(math.pow((features[i][j] - mean[1][j]), 2) / 2.0 * math.pow(std[1][j], 2))))
            for x in range(len(feature_log_prob[0])):
                if feature_log_prob[0][x] != 0:
                    ham_prob += math.log(feature_log_prob[0][x]) + math.log(class_log_prior[0])
                if feature_log_prob[1][x] != 0:
                    spam_prob += math.log(feature_log_prob[1][x]) + math.log(class_log_prior[1])
            if ham_prob > spam_prob:
                classes[i] = 0
            else:
                classes[i] = 1
        return classes