import math
import os
from collections import Counter
from pathlib import Path
from typing import List
import numpy as np

from spam_filter import Bernoulli, Gauss, MultinomialNaiveBayes



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

    train_features = generate_features(common_map, train_file_path)

    # training labels
    train_labels = np.zeros(len(files))
    train_labels[train_labels.size // 2 : train_labels.size] = 1.0
    
    files = list(test_file_path.iterdir())
    # testing feature matrix
    test_features = generate_features(common_map, test_file_path)

    # testing labels
    test_labels = np.zeros(len(files))
    c = 0
    for file in files:
        if "spm" in os.path.basename(file):
            test_labels[c] = 1
        c += 1
    #test_labels[test_labels.size // 2 : test_labels.size] = 1.0
    print("spam files: ", np.count_nonzero(test_labels))

    # ---------------------------------------------------------------------------- #
    #                                  Runner Code                                 #
    # ---------------------------------------------------------------------------- #

    # Multinomial Naive Bayes
#    Multinomial = multinomial(train_features, train_labels)
 #   classes = multinomial.predict(test_features)
  #  error = (test_labels == classes).sum()
   # print("Multinomial Naive Bayes: {:.2f}%".format(error / test_labels.shape[0] * 100))

    # Bernoulli Naive Bayes start
    BernoulliNB = Bernoulli()
    BernoulliNB.BernoulliNB(train_features, train_labels)
    classes = BernoulliNB.BernoulliNB_predict(test_features)
    errorHam = 0
    errorSpam = 0
    for i in range(len(files)):
        if test_labels[i] == 0 and test_labels[i] != classes[i]:
            errorHam += 1
        elif test_labels[i] == 1 and test_labels[i] != classes[i]:
            errorSpam += 1
    print("Bernoulli Naive Bayes Ham error: ", float(errorHam) / float(len(test_labels) - np.count_nonzero(test_labels)), "Spam Error: ", float(errorSpam) / float(np.count_nonzero(test_labels)))
    # Bernoulli Naive Bayes end

    # ---------------------------------------------------------------------------- #
    #                                  Runner Code                                 #
    # ---------------------------------------------------------------------------- #

    # Multinomial Naive Bayes
    multinomial = MultinomialNaiveBayes(train_features, train_labels)
    classes = multinomial.predict(test_features)
    error = (test_labels == classes).sum()
    print("Multinomial Naive Bayes: {:.2f}%".format(error / test_labels.shape[0] * 100))

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
    GaussianNB.GaussianNB(train_features, train_labels)
    classes = GaussianNB.GaussianNB_predict(test_features)
    error = 0
    for i in range(len(files)):
        if test_labels[i] == classes[i]:
            error += 1
    print("Gaussian Naive Bayes: ", float(error) / float(len(test_labels)))
    # Gaussian Naive Bayes end


def parse_message(text: str) -> List[str]:
    """Processes raw text message into a list of words without symbols
    Args:
        text (str): Message to be processed
    Returns:
        List[str]: array of words
    """
    word_list = text.replace("\n", " ").split(" ")
    word_list = [word for word in word_list if word != "" and word[0].isalpha()]

    return word_list


def generate_features(common_map: List[str], path: Path) -> List[List[str]]:
    """Generates a 2 dimensional feature array of shape (file, common_word)

    Args:
        common_map (List[str]): list of common words
        path (Path): a path to the directory of test files

    Returns:
        List[List[str]]: feature array of shape (file, common_word)
    """
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


if __name__ == "__main__":
    main()