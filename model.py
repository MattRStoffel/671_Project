import torch


class NeuralNetwork(torch.nn.Module):
    def __init__(self, batchSize: int, maxSeq: int, vocabSize: int):
        super().__init__()
        # Input:  (batchSize x maxSeq x vocabSize)
        self.embed = torch.nn.Embedding(vocabSize, 150)
        self.layer1 = torch.nn.Linear(150, 40)
        # output: (batchSize x maxSeq x 4)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(40, 2)
        self.layer3 = torch.nn.Linear(maxSeq, 1)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # input:  (batchSize x maxSeq x vocabSize)
        output = self.embed(x)
        output = self.layer1(output)
        output = self.relu(output)
        output = self.layer2(output)  # ( batchsize x maxSeq x 2)
        output = torch.transpose(output, 1, 2)
        output = self.layer3(output)
        output = self.softmax(output)
        return output
