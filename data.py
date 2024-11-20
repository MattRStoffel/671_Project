from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from transformers import BertTokenizer


class MyDataset(Dataset):
    def __init__(self):
        df = pd.read_csv("./ExtractedTweets.csv")
        try:
            nltk.data.find("corpora/stopwords.zip")
        except LookupError:
            nltk.download("stopwords", quiet=True)
            nltk.data.find("corpora/stopwords.zip")

        try:
            nltk.data.find("tokenizers/punkt.zip")
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.data.find("tokenizers/punkt.zip")

        # Initializing a BERT google-bert/bert-base-uncased style configuration
        self.maxSeq = 0
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.vocabSize = self.getVocabSize()
        self.stop_words = stopwords.words("english")
        self.stemmer = PorterStemmer()

        self.X = df["Tweet"].apply(self.preprocess_text)
        self.Y = df["Party"]
        print("Dataset initialized")

    def preprocess_text(self, text: str):
        # Remove URLs they dont add valuable information
        try:
            text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
            # Remove punctuation and other unwanted characters
            text = re.sub(r"[^\w\s]", "", text)
            text = text.lower()
            tokens = word_tokenize(text)
            # Remove stop words and apply stemming
            # Stemmers remove morphological affixes from words, leaving only the word stem
            tokens = [
                self.stemmer.stem(word)
                for word in tokens
                if word not in self.stop_words
            ]
            # Join tokens back to string

            text = " ".join(tokens)
            encode = self.tokenizer.encode(text, padding="max_length")
            self.maxSeq = max(self.maxSeq, len(encode))

        except Exception as e:
            print(e)
            print(text)
            exit(0)
        return self.tokenizer.encode(text, padding="max_length", max_length=self.maxSeq)

    def getVocabSize(self):
        vocabSize = self.tokenizer.vocab_size
        return vocabSize

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def get_data_loaders(batch_size: int):
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

    vocabSize = dataset.getVocabSize()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=24, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=24, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=24, pin_memory=True)
    
    return dataset.maxSeq, vocabSize, train_loader, test_loader, validation_loader
