# import math
from collections import Counter
from pathlib import Path
from typing import List, Set
import numpy as np

from spam_filter import Bernoulli, Gauss, Multinomial


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


def generate_features(path: Path, common_words: List[str]) -> np.ndarray[np.float64]:
    """Generates 2D array of where each row is an individual file,
    and each column is score for the occurance of the specific word

    Args:
        path (Path): file path
        common_words (Set[str]): List of common words 

    Returns:
        [type]: [description]
    """
    files = list(path.iterdir())
    features = np.zeros((len(files), len(common_words)))
    # file_index = 0

    for file in files:
        text = file.read_text()
        word_list = parse_message(text)
        file_counter = Counter(word_list)

        common_index = 0
        for word in common_words:

            if word in file_counter:
                # ipdb.set_trace()
                features[files.index(file)][common_index] = file_counter[word]
            common_index += 1
        # file_index += 1
    return features


def main():
    test_file_path = Path("test-mails")
    train_file_path = Path("train-mails")

    word_counter = Counter()
    files = list(train_file_path.iterdir())

    for file in files:
        text = file.read_text()
        word_list = parse_message(text)
        word_counter += Counter(word_list)

    print("The maximum of most_common can be: ", len(word_counter))
    common_words = [k for k, _ in word_counter.most_common(3000)]

    train_features = generate_features(train_file_path, common_words)

    train_labels = np.zeros(len(files))
    train_labels[train_labels.size // 2 : train_labels.size] = 1.0

    files = list(test_file_path.iterdir())
    test_features = generate_features(test_file_path, common_words)

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
