import torch
import torch.nn.functional as F
from data import get_data_loaders
import pickle
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class NeuralNetwork(torch.nn.Module):
    def __init__(self, batchSize: int, maxSeq: int, vocabSize: int):
        super().__init__()
        # Input:  (batchSize x maxSeq x vocabSize)
        self.embed = torch.nn.Embedding(vocabSize, 150)
        self.layer1 = torch.nn.Linear(150, 40)
        # output: (batchSize x maxSeq x 4)
        self.layer2 = torch.nn.Linear(40, 2)
        self.laye3 = torch.nn.Linear(maxSeq, 1)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # input:  (batchSize x maxSeq x vocabSize)
        output = self.embed(x)
        output = self.layer1(output)
        output = self.layer2(output)  # ( batchsize x maxSeq x 2)
        output = torch.transpose(output, 1, 2)
        output = self.laye3(output)
        output = self.softmax(output)
        return output


def definingLabel(label: str):
    if label.lower() == "republican":
        y = [0.0, 1.0]
    else:
        y = [1.0, 0.0]
    return torch.tensor(y)


def graph_accuracy(accuracy):
    plt.plot(accuracy)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()


def save_model(model, epoch):
    with open(f"model_{epoch}.pkl", "wb") as f:
        pickle.dump(model, f)


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    for X, y in train_loader:
        X = torch.stack(X, dim=1).int().to(device)
        listOfLabels = [definingLabel(label) for label in y]
        listOfLabels = torch.stack(listOfLabels, dim=0).int().to(device)
        pred = model.forward(X).squeeze(dim=2)
        loss = F.cross_entropy(pred.float(), listOfLabels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(model, validation_loader, device):
    model.eval()
    correct_count = 0
    incorrect_count = 0
    with torch.no_grad():
        for X, y in validation_loader:
            X = torch.stack(X, dim=1).int().to(device)
            listOfLabels = [definingLabel(label) for label in y]
            listOfLabels = torch.stack(listOfLabels, dim=0).int().to(device)
            pred = model.forward(X).squeeze(dim=2)
            predVal = pred.argmax(dim=1)
            actVal = listOfLabels.argmax(dim=1)
            correct_count += (predVal == actVal).sum().item()
            incorrect_count += (predVal != actVal).sum().item()
    return correct_count / (correct_count + incorrect_count)


def train():
    batchsize = 2
    maxSeq, vocabSize, train_loader, _, validation_loader = get_data_loaders(batchsize)
    model = NeuralNetwork(batchSize=batchsize, maxSeq=maxSeq, vocabSize=vocabSize)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = "cpu"
    epochs = 20
    validation_accuracy = []

    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, device)
        validation_acc = validate(model, validation_loader, device)
        validation_accuracy.append(validation_acc)
        print(f"Epoch {epoch}, Validation Accuracy: {validation_acc}")
        save_model(model, epoch)

    graph_accuracy(validation_accuracy)


train()
