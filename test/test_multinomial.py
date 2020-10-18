import unittest, numpy as np
from spam_filter import MultinomialNaiveBayes
from collections import Counter
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB

def parse_message(text: str) -> List[str]:
    word_list = text.replace("\n", " ").split(" ")
    word_list = [word for word in word_list if word.isalpha()]

    return word_list

def generate_features(path: Path, common_words: Set[str]):
    files = list(path.iterdir())
    features = np.zeros((len(files), len(common_words)))

    for file in files:
        text = file.read_text()
        word_list = parse_message(text)
        file_counter = Counter(word_list)

        common_index = 0
        for word in file_counter:
            if word in common_words:
                features[files.index(file)][common_index] = file_counter[word]

            common_index += 1

    return features

def runTest(self):
    # Reused code from main.py
    test_file_path = Path("../test-mails")
    train_file_path = Path("../train-mails")

    word_counter = Counter()
    files = list(train_file_path.iterdir())

    for file in files:
        text = file.read_text()
        word_list = parse_message(text)
        word_counter += Counter(word_list)

    common_words = {k for k, _ in word_counter.most_common(3000)}

    train_features = generate_features(train_file_path, common_words)
    train_labels = np.zeros(len(files))

    train_labels[train_labels.size // 2 : train_labels.size] = 1.0

    files = list(test_file_path.iterdir())
    test_features = generate_features(test_file_path, common_words)

    test_labels = np.zeros(len(files))
    test_labels[test_labels.size // 2 : test_labels.size] = 1.0

    multinomial = MultinomialNaiveBayes()
    multinomial.MultinomialNB(train_features, train_labels)
    classes = multinomial.MultinomialNB_predict(test_features)
    error = (test_labels == classes).sum()

    multinomial_sklearn = MultinomialNB()
    multinomial_sklearn.fit(train_features, train_labels)
    classes_sklearn = multinomial.predict(test_features)
    error_sklearn = (test_labels == classes_sklearn).sum()

    assert(error == error_sklearn, "Error values are not the same. Our error: " + error + ", SKLearn's error: " + error_sklearn)

if __name__ == "__main__":
    runTest()