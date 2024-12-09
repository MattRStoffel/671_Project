import torch
import torch.nn.functional as F
from model import NeuralNetwork
import data
import util
from util import get_cpu_info
cpu_name, num_threads = get_cpu_info()

device = util.get_device()


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X = torch.stack(X, dim=1).int().to(device)
        listOfLabels = [label for label in y]
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
            X = torch.stack(X, dim=1).int().to(device)
            listOfLabels = [label for label in y]
            listOfLabels = torch.stack(listOfLabels, dim=0).int()
            pred = model(X).squeeze(dim=2)
            predVal = pred.argmax(dim=1)
            actVal = listOfLabels.argmax(dim=1)
            correct_count += (predVal == actVal).sum().item()
            incorrect_count += (predVal != actVal).sum().item()
    return correct_count / (correct_count + incorrect_count)


def train(epochs=5, batchsize=200, learning_rate=0.001):
    maxSeq, vocabSize, train_loader, _, validation_loader = data.get_data_loaders(
        batchsize
    )
    model = NeuralNetwork(batchSize=batchsize, maxSeq=maxSeq, vocabSize=vocabSize).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    validation_accuracy = []
    trainings_loss = []

    for epoch in range(epochs):
        average_loss = train_one_epoch(model, train_loader, optimizer, device)
        trainings_loss.append(average_loss)
        validation_acc = validate(model, validation_loader, device)
        validation_accuracy.append(validation_acc)
        print(
            f"Epoch {epoch + 1}, Loss: {average_loss:.5f}, Validation Accuracy: {validation_acc:.5f}"
        )

    return model, trainings_loss, validation_accuracy


def my_grid_search():
    epochs = [4]
    batchsizes = [500]
    learning_rates = [0.001]
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
                util.save_results(results, "results.pkl")

    # print top 5 models
    results = sorted(results.items(), key=lambda x: x[1][2], reverse=True)
    for i, (params, (_, _, validation_accuracy)) in enumerate(results):
        print(f"Model {i+1}: {params}, Validation Accuracy: {validation_accuracy[-1]}")


if __name__ == "__main__":
    print(f"Using {device} device")
    if device == "cuda":
        print("Is CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print(
            "Device name:",
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        )
        print(f"Detected CPU: {cpu_name}")
        print(f"Setting number of CPU threads: {num_threads}")
    my_grid_search()
