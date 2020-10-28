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
    return np.full(NUM_CLASSES, -np.log(NUM_CLASSES))


from .bernoulli import BernoulliNB
from .gauss import GaussianNB
from .multinomial import MultinomialNB
