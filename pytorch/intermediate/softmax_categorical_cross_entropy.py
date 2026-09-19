import torch
from torch import nn

'''
for softmax, the labels must be one-hot encoded for the Categorical Cross Entropy to
work

the torch.nn.CrossEntropyLoss works off of the logits, so the values it accepts are from
the linear weighted average layer not, the softmax probability layer
'''

# loss = nn.CrossEntropyLoss()
#
# Y = torch.tensor([0, 2])
# # inputs of n_samples, n_classes
# Y_pred_good = torch.tensor([[2.0, 1.0, 0.1],
#                             [0.3, 0.61, 1]])
# Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3],
#                            [2.3, 0.3, 0.1]])
#
# l1 = loss(Y_pred_good, Y)
# l2 = loss(Y_pred_bad, Y)
#
# print(l1.item())
# print(l2.item())
#
# _, prediction1 = torch.max(Y_pred_good, dim=1)
# _, prediction2 = torch.max(Y_pred_bad, dim=1)
#
# print(prediction1)
# print(prediction2)

class NeuralNet2(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(x)
        # no softmax because nn.CrossEntropyLoss() uses logits
        return out

model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()

