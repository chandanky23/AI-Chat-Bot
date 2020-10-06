import torch
import torch.nn as nn

# This model uses a feed Forward neural net pattern
class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNet, self).__init__()
    self.l1 = nn.Linear(input_size, hidden_size)
    self.l2 = nn.Linear(hidden_size, hidden_size)
    self.l3 = nn.Linear(hidden_size, num_classes) # input_size and num_classes should be fixed but hidden_size can change
    self.relu = nn.ReLU()

  def forward(self, x):
    out = self.l1(x)
    out = self.relu(out)
    out = self.l2(out)
    out = self.relu(out)
    out = self.l3(out)

    # no activation and no softmax as we later apply the crossentropy loss and it will apply this activation internally.
    return out
  