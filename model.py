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

class NeuralNetwork(torch.nn.Module):
    def __init__(self, batchSize: int, maxSeq: int, vocabSize: int):
        super().__init__()
        # Input:  (batchSize x maxSeq x vocabSize)
        self.embed = torch.nn.Embedding(vocabSize, 150)
        self.layer1 = torch.nn.Linear(150, 40)
        # output: (batchSize x maxSeq x 4)
        self.layer2 = torch.nn.Linear(40, 2)
        self.layer3 = torch.nn.Linear(maxSeq, 1)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # input:  (batchSize x maxSeq x vocabSize)
        output = self.embed(x)
        output = self.layer1(output)
        output = self.layer2(output)  # ( batchsize x maxSeq x 2)
        output = torch.transpose(output, 1, 2)
        output = self.layer3(output)
        output = self.softmax(output)
        return output


def definingLabel(label: str):
    if label.lower() == "republican":
        y = [0.0, 1.0]
    else:
        y = [1.0, 0.0]
    return torch.tensor(y)


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X = torch.stack(X, dim=1).int().to(device)
        listOfLabels = [definingLabel(label).to(device) for label in y]
        listOfLabels = torch.stack(listOfLabels, dim=0).int()
        pred = model(X).squeeze(dim=2)
        loss = F.cross_entropy(pred.float(), listOfLabels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate(model, validation_loader, device):
    model.eval()
    correct_count = 0
    incorrect_count = 0
    with torch.no_grad():
        for X, y in validation_loader:
            X = torch.stack(X, dim=1).int().to(device)  # Move input data to the correct device
            listOfLabels = [definingLabel(label).to(device) for label in y]  # Move labels to the device
            listOfLabels = torch.stack(listOfLabels, dim=0).int()
            pred = model(X).squeeze(dim=2)
            predVal = pred.argmax(dim=1)
            actVal = listOfLabels.argmax(dim=1)
            correct_count += (predVal == actVal).sum().item()
            incorrect_count += (predVal != actVal).sum().item()
    return correct_count / (correct_count + incorrect_count)


def train(epochs=5, batchsize=200, learning_rate=0.001):
    maxSeq, vocabSize, train_loader, _, validation_loader = get_data_loaders(batchsize)
    model = NeuralNetwork(batchSize=batchsize, maxSeq=maxSeq, vocabSize=vocabSize).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    validation_accuracy = []
    trainings_loss = []

    for epoch in range(epochs):
        average_loss = train_one_epoch(model, train_loader, optimizer, device)
        trainings_loss.append(average_loss)
        validation_acc = validate(model, validation_loader, device)
        validation_accuracy.append(validation_acc)
        print(
            f"Epoch {epoch}, Loss: {average_loss:.5f}, Validation Accuracy: {validation_acc:.5f}"
        )

    return model, trainings_loss, validation_accuracy


def my_grid_search():
    epochs = [4, 8, 16, 32]
    batchsizes = [64, 128, 256, 512]
    learning_rates = [0.001, 0.01, 0.1]
    results = {}
    for epoch in epochs:
        for batchsize in batchsizes:
            for learning_rate in learning_rates:
                print(
                    f"Training with epochs: {epoch}, batchsize: {batchsize}, learning_rate: {learning_rate}"
                )
                model, trainings_loss, validation_accuracy = train(
                    epoch, batchsize, learning_rate
                )
                results[(epoch, batchsize, learning_rate)] = (
                    model,
                    trainings_loss,
                    validation_accuracy,
                )

    # print top 5 models
    results = sorted(results.items(), key=lambda x: x[1][2], reverse=True)
    for i, (params, (_, _, validation_accuracy)) in enumerate(results):
        print(f"Model {i+1}: {params}, Validation Accuracy: {validation_accuracy[-1]}")

    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    print(f"Using {device} device")
    print("Is CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

    my_grid_search()
