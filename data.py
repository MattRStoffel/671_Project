from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


class MyDataset(Dataset):
    def __init__(self):
        df = pd.read_csv("./ExtractedTweets.csv")
        try:
            nltk.data.find("corpora/stopwords.zip")
        except LookupError:
            nltk.download("stopwords", quiet=True)

        try:
            nltk.data.find("tokenizers/punkt.zip")
        except LookupError:
            nltk.download("punkt", quiet=True)

        self.stop_words = stopwords.words("english")
        self.stemmer = PorterStemmer()
        self.X = df["Tweet"].apply(self.preprocess_text)
        self.Y = df["Party"]

    def preprocess_text(self, text):
        # Remove URLs they dont add valuable information
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        # Remove punctuation and other unwanted characters
        text = re.sub(r"[^\w\s]", "", text)
        text = text.lower()
        tokens = word_tokenize(text)
        # Remove stop words and apply stemming
        # Stemmers remove morphological affixes from words, leaving only the word stem
        tokens = [
            self.stemmer.stem(word) for word in tokens if word not in self.stop_words
        ]
        # Join tokens back to string
        text = " ".join(tokens)
        return text

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def get_data_loaders():
    dataset = MyDataset()
    train_size = int(0.8 * len(dataset))
    test_size = int(0.8 * (len(dataset) - train_size))
    validate_size = len(dataset) - train_size - test_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size + validate_size]
    )
    test_dataset, validation_dataset = random_split(
        test_dataset, [test_size, validate_size]
    )

    batch_sizes = 3
    # Create data loaders for the train and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_sizes)
    test_loader = DataLoader(test_dataset, batch_size=batch_sizes)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_sizes)

    return train_loader, test_loader, validation_loader
