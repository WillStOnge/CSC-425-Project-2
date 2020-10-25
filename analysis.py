import math, os, time, numpy as np
from collections import Counter
from pathlib import Path
from typing import List
from spam_filter import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from main import parse_message, generate_features


def main():
    test_file_path = Path("test-mails")
    train_file_path = Path("train-mails")
    files = list(train_file_path.iterdir())

    word_counter = Counter()
    for file in files:
        text = file.read_text()
        word_list = parse_message(text)
        word_counter += Counter(word_list)

    common_map = [w for w, _ in word_counter.most_common(3000)]
    train_features = generate_feature(common_map, train_file_path)
    train_labels = np.zeros(len(files))
    train_labels[train_labels.size // 2:train_labels.size] = 1.0
    files = list(test_file_path.iterdir())
    test_features = generate_feature(common_map, test_file_path)
    test_labels = np.zeros(len(files))
    test_labels[test_labels.size // 2:test_labels.size] = 1.0

    start_time = time.time()

    # Multinomial Naive Bayes
    multinomial = MultinomialNB(train_features, train_labels)
    multinomial.predict(test_features)

    print("Multinomial execution time: {:.2}".format(time.time() - start_time))
    start_time = time.time()

    # Bernoulli Naive Bayes
    bernoulli = BernoulliNB(train_features, train_labels)
    bernoulli.predict(test_features)

    print("Bernoulli execution time: {:.2}".format(time.time() - start_time))
    start_time = time.time()

    # Gaussian Naive Bayes
    gaussian = GaussianNB(train_features, train_labels)
    gaussian.predict(test_features)

    print("Gaussian execution time: {:.2}".format(time.time() - start_time))
    start_time = time.time()

    # SKLearn Multinomial Naive Bayes
    multinomial_sklearn = MultinomialNB()
    multinomial_sklearn.fit(train_features, train_labels)
    classes_sklearn = multinomial_sklearn.predict(test_features)

    print("SKLearn Multinomial execution time: {:.2}".format(time.time() - start_time))
    start_time = time.time()

    # SKLearn Bernoulli Naive Bayes
    bernoulli_sklearn = BernoulliNB()
    bernoulli_sklearn.fit(train_features, train_labels)
    bernoulli_sklearn.predict(test_features)

    print("SKLearn Bernoulli execution time: {:.2}".format(time.time() - start_time))
    start_time = time.time()

    # SKLearn Gaussian Naive Bayes
    gaussian_sklearn = GaussianNB()
    gaussian_sklearn.fit(train_features, train_labels)
    gaussian_sklearn.predict(test_features)

    print("SKLearn Gaussian execution time: {:.2}".format(time.time() - start_time))


if __name__ == "__main__":
    main()