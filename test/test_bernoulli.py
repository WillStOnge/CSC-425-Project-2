import unittest, numpy as np
from pathlib import Path
from collections import Counter
from spam_filter import BernoulliNB as BernoulliNaiveBayes
from sklearn.naive_bayes import BernoulliNB
from main import parse_message, generate_features
import ipdb


class TestBernoulliNB(unittest.TestCase):
    def setUp(self):
        test_file_path = Path("test-mails")
        train_file_path = Path("train-mails")

        files = list(train_file_path.iterdir())

        word_counter = Counter()
        for file in files:
            text = file.read_text()
            word_list = parse_message(text)
            word_counter += Counter(word_list)

        common_words = {k for k, _ in word_counter.most_common(3000)}

        self.train_features = generate_features(common_words, train_file_path)
        self.train_labels = np.zeros(len(files))
        self.train_labels[self.train_labels.size // 2 : self.train_labels.size] = 1.0

        files = list(test_file_path.iterdir())

        self.test_features = generate_features(common_words, test_file_path)
        self.test_labels = np.zeros(len(files))
        self.test_labels[self.test_labels.size // 2 : self.test_labels.size] = 1.0

    def test_predict(self):
        our_bnb = BernoulliNaiveBayes(self.train_features, self.train_labels)
        our_classes = our_bnb.predict(self.test_features)
        our_error = (self.test_labels == our_classes).sum()

        their_bnb = BernoulliNB()
        their_bnb.fit(self.train_features, self.train_labels)
        their_classes = their_bnb.predict(self.test_features)
        their_error = (self.test_labels == their_classes).sum()

