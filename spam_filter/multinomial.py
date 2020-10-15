import os
import math
import numpy as np
import enum

MOST_COMMON_WORD = 3000
SMOOTH_ALPHA = 1.0
CLASS_NUM = len(Classification)

class_log_prior = [0.0, 0.0]
feature_log_prob = np.zeros((CLASS_NUM, MOST_COMMON_WORD))

class Classification(enum):
	HAM = 0
	SPAM = 1

class MultinomialNaiveBayes:
    """ Multinomial Naive Bayes """
    def __init__(self, features, labels):
		""" Constructor that trains the model using Multinomial NB """

		class_log_prior[Classification.HAM] = math.log(np.sum(labels == Classification.HAM))
		class_log_prior[Classification.SPAM] = math.log(np.sum(labels == Classification.SPAM))
		
		# Calculate feature_log_prob
		ham_words = list()
		spam_words = list()
		ham_sum = 0
		spam_sum = 0

		# Nested loop over features.
		for i in range(features.length):
			for j in range(MOST_COMMON_WORD):
				ham_words[j] += features[i][j]
				spam_words[j] += features[i][j]
				ham_sum += features[i][j]
				spam_sum += features[i][j]

		# Add smooth alpha value to each item in the words lists.
		for i in range(MOST_COMMON_WORD):
			ham_words[i] += SMOOTH_ALPHA
			spam_words[i] += SMOOTH_ALPHA
		ham_sum += MOST_COMMON_WORD * SMOOTH_ALPHA
		spam_sum += MOST_COMMON_WORD * SMOOTH_ALPHA

		for i in range(MOST_COMMON_WORD):
			feature_log_prob[Classification.HAM][i] = math.log(ham_words[i] / ham_sum)
			feature_log_prob[Classification.SPAM][i] = math.log(spam_words[i] / spam_sum)
		 
    def predict(self, features):
		""" Classify an email's feature vector. """
		classes = np.zeros(len(features))

		for i in range(features.length):
			ham_prob = 0
			spam_prob = 0
			for j in range(features[i].length):
				# TODO: Calculate ham_prob and spam_prob ¯\_(ツ)_/¯
				ham_prob += class_log_prior[Classification.HAM]
				spam_prob += class_log_prior[Classification.SPAM]
			classes[i] = Classification.HAM if ham_prob > spam_prob else Classification.SPAM
		
		return classes
