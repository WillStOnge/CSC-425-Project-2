# import math
from collections import Counter
from pathlib import Path
from typing import List, Set
import numpy as np

from spam_filter import Bernoulli, Gauss, Multinomial

test_file_path = Path("test-mails")
train_file_path = Path("train-mails")


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


def generate_feature(path: Path, common_map: Set[str]):
    files = list(path.iterdir())
    features = np.zeros((len(files), len(common_map)))
    # file_index = 0

    for file in files:
        text = file.read_text()
        word_list = parse_message(text)
        file_counter = Counter(word_list)

        common_index = 0
        for word in common_map:

            if word in file_counter:
                # ipdb.set_trace()
                features[files.index(file)][common_index] = file_counter[word]
            common_index += 1
        # file_index += 1
    return features


def main():
    word_counter = Counter()
    files = list(train_file_path.iterdir())

    for file in files:
        text = file.read_text()
        word_list = parse_message(text)
        word_counter += Counter(word_list)

    print("The maximum of most_common can be: ", len(word_counter))
    common_words = [k for k, _ in word_counter.most_common(3000)]

    # construct model
    # training feature matrix

    # features = np.zeros((len(files), len(common_words)))

    train_features = generate_feature(train_file_path, common_words)

    train_labels = np.zeros(len(files))
    train_labels[train_labels.size // 2 : train_labels.size] = 1.0

    # verify model
    # load test data
    files = list(test_file_path.iterdir())
    # testing feature matrix
    test_features = generate_feature(test_file_path, common_words)

    # testing labels
    test_labels = np.zeros(len(files))
    test_labels[test_labels.size // 2 : test_labels.size] = 1.0

    # Multinomial Naive Bayes start
    multinomial = Multinomial()
    multinomial.MultinomialNB(train_features, train_labels)
    classes = multinomial.MultinomialNB_predict(test_features)

    error = (test_labels == classes).sum()

    print("Multinomial Naive Bayes: ", float(error) / float(len(test_labels)))

    # # Bernoulli Naive Bayes start
    BernoulliNB = Bernoulli()
    BernoulliNB.BernoulliNB(train_features, train_labels)
    classes = BernoulliNB.BernoulliNB_predict(test_features)

    error = (test_labels == classes).sum()

    print("Bernoulli Naive Bayes: ", float(error) / float(len(test_labels)))

    # Gaussian Naive Bayes start
    GaussianNB = Gauss()
    GaussianNB.GaussianNB(train_features, train_labels)
    classes = GaussianNB.GaussianNB_predict(test_features)

    error = (test_labels == classes).sum()

    print("Gaussian Naive Bayes: ", float(error) / float(len(test_labels)))


if __name__ == "__main__":
    main()
