import math
import os
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np

from spam_filter import Bernoulli, Gauss, Multinomial

# read file anmes in the specific file path


def parse_message(text: str) -> List[str]:
    """Processes raw text message into a list of words without symbols
    Args:
        text (str): Message to be processed
    Returns:
        List[str]: array of words
    """
    word_list = text.replace("\n", " ").split(" ")
    word_list = [word for word in word_list if word.isalpha()]

    return word_list


# generate features according to commonMap
def generate_feature(common_map: List[str], path: Path) -> List[List[str]]:
    files = list(path.iterdir())
    dimensions = (len(files), len(common_map))
    features = np.zeros(dimensions)

    file_index = 0
    for file in files:
        text = file.read_text()
        word_list = parse_message(text)
        word_counter = Counter(word_list)

        common_index = 0
        for key in common_map:
            if key in word_counter:
                features[file_index][common_index] = word_counter[key]

            common_index += 1
        file_index += 1
    return features


def main():

    test_file_path = Path("test-mails")
    train_file_path = Path("train-mails")
    files = list(train_file_path.iterdir())

    # ---------------------------------------------------------------------------- #
    #                        Count total words per all files                       #
    # ---------------------------------------------------------------------------- #

    word_counter = Counter()
    for file in files:
        text = file.read_text()
        word_list = parse_message(text)
        word_counter += Counter(word_list)

    print(f"The maximum of most_common can be: {len(word_counter)}")

    # ---------------------------------------------------------------------------- #
    #                               Compute common                                 #
    # ---------------------------------------------------------------------------- #

    common_map = [w for w, _ in word_counter.most_common(3000)]

    train_features = generate_feature(common_map, train_file_path)

    # training labels
    train_labels = np.zeros(len(files))
    train_labels[train_labels.size // 2:train_labels.size] = 1.0

    files = list(test_file_path.iterdir())
    # testing feature matrix
    test_features = generate_feature(common_map, test_file_path)

    # testing labels
    test_labels = np.zeros(len(files))
    test_labels[test_labels.size // 2:test_labels.size] = 1.0

    # ---------------------------------------------------------------------------- #
    #                                  Runner Code                                 #
    # ---------------------------------------------------------------------------- #

    MultinomialNB = Multinomial()
    MultinomialNB.MultinomialNB(train_features, train_labels)
    # test model
    classes = MultinomialNB.MultinomialNB_predict(test_features)
    error = 0
    for i in range(len(files)):
        if test_labels[i] == classes[i]:
            error += 1
    print("Multinomial Naive Bayes: ", float(error) / float(len(test_labels)))
    # Multinomial Naive Bayes end

    # Bernoulli Naive Bayes start
    BernoulliNB = Bernoulli()
    BernoulliNB.BernoulliNB(train_features, train_labels)
    classes = BernoulliNB.BernoulliNB_predict(test_features)
    error = 0
    for i in range(len(files)):
        if test_labels[i] == classes[i]:
            error += 1
    print("Bernoulli Naive Bayes: ", float(error) / float(len(test_labels)))
    # Bernoulli Naive Bayes end

    # Gaussian Naive Bayes start
    GaussianNB = Gauss()
    return GaussianNB.GaussianNB(train_features, train_labels)
    classes = GaussianNB.GaussianNB_predict(test_features)
    # error = 0
    # for i in range(len(files)):
    #     if test_labels[i] == classes[i]:
    #         error += 1
    # print("Gaussian Naive Bayes: ", float(error) / float(len(test_labels)))
    # Gaussian Naive Bayes end


if __name__ == "__main__":
    main()
