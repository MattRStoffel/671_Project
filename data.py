from torch import tensor
from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from transformers import BertTokenizer
import pickle
import os
import util
from util import get_cpu_info


class MyDataset(Dataset):
    def __init__(self):
        preprocessed_file = "preprocessed_data.pkl"
        if not os.path.exists(preprocessed_file):
            preprocess_and_save("ExtractedTweets.csv", preprocessed_file)

        with open(preprocessed_file, "rb") as f:
            self.data, self.vocabSize, self.maxSeq = pickle.load(f)

    def __len__(self):
        return len(self.data["tweets"])

    def __getitem__(self, idx):
        tweet = self.data["tweets"][idx]
        label = self.data["labels"][idx]
        return tweet, label

def preprocess_text(text):
    nltk.data.find("corpora/stopwords.zip")

    nltk.data.find("tokenizers/punkt.zip")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    stop_words = stopwords.words("english")
    stemmer = PorterStemmer()
    maxSeq = 128
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    text = " ".join(tokens)
    return tokenizer.encode(text, padding="max_length", max_length=maxSeq)

def preprocess_and_save(csv_file, preprocessed_file):
    df = pd.read_csv(csv_file)


    tweets = df["Tweet"].apply(preprocess_text).tolist()
    labels = df["Party"].apply(definingLabel).tolist()

    data = ({"tweets": tweets, "labels": labels}, tokenizer.vocab_size, maxSeq)
    with open(preprocessed_file, "wb") as f:
        pickle.dump(data, f)


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

    num_workers = 1

    vocabSize = dataset.vocabSize
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataset.maxSeq, vocabSize, train_loader, test_loader, validation_loader


def definingLabel(label: str):
    if label.lower() == "republican":
        y = [0.0, 1.0]
    else:
        y = [1.0, 0.0]
    return tensor(y).to(util.get_device())
