import os, math, enum
import numpy as np

class Classification(enum.Enum):
	HAM = 0
	SPAM = 1

NUM_CLASSES = len(Classification)
MOST_COMMON_WORD = 3000
SMOOTH_ALPHA = 1.0

class MultinomialNaiveBayes:
	class_log_prior = np.zeros(NUM_CLASSES, dtype=np.float64)
	feature_log_prob = np.zeros((NUM_CLASSES, MOST_COMMON_WORD))

	""" Multinomial Naive Bayes """
	def __init__(self, features: np.array, labels: np.array):
		""" Constructor that trains the model using Multinomial NB """
		self.class_log_prior[Classification.HAM] = math.log(np.sum(labels == Classification.HAM))
		self.class_log_prior[Classification.SPAM] = math.log(np.sum(labels == Classification.SPAM))
		
		# Calculate feature_log_prob
		ham_words = list()
		spam_words = list()
		ham_sum = np.sum(features)
		spam_sum = np.sum(features)

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
			self.feature_log_prob[Classification.HAM][i] = math.log(ham_words[i] / ham_sum)
			self.feature_log_prob[Classification.SPAM][i] = math.log(spam_words[i] / spam_sum)
		
	def predict(self, features: np.array):
		""" Classify an email's feature vector. """
		classes = np.zeros(len(features))

		for i in range(features.length):
			ham_prob = 0
			spam_prob = 0
			for j in range(features[i].length):
				smooth_features = features + SMOOTH_ALPHA
				smooth_features_sum = smooth_features.sum(axis=1)

				self.feature_log_prob = np.log(smooth_features) - np.log(smooth_features_sum.reshape(-1, 1))

				ham_prob += self.class_log_prior[Classification.HAM]
				spam_prob += self.class_log_prior[Classification.SPAM]
			classes[i] = Classification.HAM if ham_prob > spam_prob else Classification.SPAM
		
		return classes
