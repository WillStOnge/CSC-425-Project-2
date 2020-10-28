import math, numpy as np
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
        # Convert features to l0-norm
        self.features = features
        self.features[features != 0] = 1

        # Count of each feature, same as co
        self.class_log_prior = generate_class_log_prior(labels)

        self.class_count = (
            np.array([(labels == 0).sum(), (labels == 1).sum()]) + SMOOTH_ALPHA * 2
        )

        fixed_labels = np.array([1 - labels, labels])
        self.feature_count = np.zeros((NUM_CLASSES, MOST_COMMON_WORD), dtype=np.float64)
        self.feature_count += np.dot(fixed_labels, self.features) + SMOOTH_ALPHA

        self.feature_log_prob = np.log(self.feature_count) - np.log(
            self.class_count.reshape(-1, 1)
        )

    def predict(self, features):
        fixed_features = features
        fixed_features[features != 0] = 1

        negative_p = np.log(1 - np.exp(self.feature_log_prob))

        log_prob = (
            np.dot(fixed_features, (self.feature_log_prob - negative_p).T)
            + self.class_log_prior
            + negative_p.sum(axis=1)
        )

        classes = np.array([Classification.HAM.value, Classification.SPAM.value])
        return classes[np.argmax(log_prob, axis=1)]
