from enum import Enum
import numpy as np
import math


class Classification(Enum):
    HAM = 0
    SPAM = 1


NUM_CLASSES = len(Classification)
MOST_COMMON_WORD = 3000
SMOOTH_ALPHA = 1.0


def generate_class_log_prior(labels: np.ndarray):
    class_log_prior = np.zeros(NUM_CLASSES, dtype=np.float64)

    class_log_prior[Classification.HAM.value] = math.log(
        np.sum(labels == Classification.HAM.value)
    )
    class_log_prior[Classification.SPAM.value] = math.log(
        np.sum(labels == Classification.SPAM.value)
    )

    return class_log_prior


from .bernoulli import BernoulliNB
from .gauss import GaussianNB
from .multinomial import MultinomialNB

