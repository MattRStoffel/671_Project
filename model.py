import torch
import torch.nn.functional as F
from data import get_data_loaders

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
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


# Access maxSeq and vocabSize
batchsize = 3
maxSeq, vocabSize, train_loader, test_loader, validation_loader = get_data_loaders(batchsize)
model = NeuralNetwork(batchSize=batchsize, maxSeq=maxSeq, vocabSize=vocabSize)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = 'cpu'

#Train model
epochs = 500
for epoch in range(epochs):
    correct_count = 0
    incorrect_count = 0
    model = model.train()
    for X, y in train_loader:
        X = torch.stack(X, dim=1).int().to(device)
        listOfLabels = []
        for label in y:
            listOfLabels.append(definingLabel(label))
        listOfLabels = torch.stack(listOfLabels, dim=0).int().to(device)
        pred = model.forward(X)
        pred = pred.squeeze(dim=2)
        #****************DELETE (debug) *****************************************
        predVal = pred.argmax(dim=1)
        actVal = listOfLabels.argmax(dim=1)
        isCorrect = (predVal == actVal)
        isWrong = (predVal != actVal)
        correct_count = correct_count + isCorrect.sum().item()
        incorrect_count = incorrect_count + isWrong.sum().item()
        # ****************DELETE (debug) *****************************************
        loss = F.cross_entropy(pred.float(), listOfLabels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("model correct: ", correct_count)
    print("model incorrect: ", incorrect_count)

#Testing
#for X, y in test_loader:
#    X = torch.stack(X, dim=1).int().to(device)
#    listOfLabels = []
#    for label in y:
#       listOfLabels.append(definingLabel(label))
#        listOfLabels = torch.stack(listOfLabels, dim=0).int().to(device)
#        pred = model.forward(X)
#        pred = pred.squeeze(dim=2)
#        loss = F.cross_entropy(pred.float(), listOfLabels.float())
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
