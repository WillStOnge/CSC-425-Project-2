import math, enum, numpy as np

class Classification(enum.Enum):
	HAM = 0
	SPAM = 1

NUM_CLASSES = len(Classification)
MOST_COMMON_WORD = 3000
SMOOTH_ALPHA = 1.0

class MultinomialNaiveBayes:
	""" Multinomial Naive Bayes """
	class_log_prior = np.zeros(NUM_CLASSES, dtype=np.float64)
	feature_log_prob = np.zeros((NUM_CLASSES, MOST_COMMON_WORD))

	def __init__(self, features: np.array, labels: np.array):
		""" Trains the model using Multinomial NB. """
		self.class_log_prior[Classification.HAM] = math.log(np.sum(labels == Classification.HAM))
		self.class_log_prior[Classification.SPAM] = math.log(np.sum(labels == Classification.SPAM))
		
		# Instructions said these are seperate, but the values are the same?
		ham_words = np.sum(features, axis=1)
		spam_words = np.sum(features, axis=1)

		# Instructions said these are seperate, but the values are the same?
		ham_sum = np.sum(features) + (MOST_COMMON_WORD * SMOOTH_ALPHA)
		spam_sum = np.sum(features) + (MOST_COMMON_WORD * SMOOTH_ALPHA)

		# Apply smoothing value to each element in the list.
		for i in range(features.length):
			ham_words[i] += SMOOTH_ALPHA
			spam_words[i] += SMOOTH_ALPHA

		# Setup feature_log_prob (this is what the prediction algorithm uses).
		for i in range(MOST_COMMON_WORD):
			self.feature_log_prob[Classification.HAM][i] = math.log(ham_words[i] / ham_sum)
			self.feature_log_prob[Classification.SPAM][i] = math.log(spam_words[i] / spam_sum)
		
	def predict(self, features: np.array):
		""" Classify an array of emails feature vectors. """
		classes = np.zeros(len(features))

		for i in range(features.length):
			ham_prob = 0
			spam_prob = 0

			# Calculate the probability that the email is spam or ham.
			for j in range(features[i].length):
				smooth_features = features + SMOOTH_ALPHA
				smooth_features_sum = smooth_features.sum(axis=1)

				self.feature_log_prob = np.log(smooth_features) - np.log(smooth_features_sum.reshape(-1, 1))

				ham_prob += self.class_log_prior[Classification.HAM]
				spam_prob += self.class_log_prior[Classification.SPAM]

			# Determine which probability is higher.
			classes[i] = Classification.HAM if ham_prob > spam_prob else Classification.SPAM
		
		return classes
