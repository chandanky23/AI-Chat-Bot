import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('intents.json', 'r') as f:
  intents = json.load(f)

all_words = []
tags = []
xy = [] # Holds our patterns and tags together

for intent in intents['intents']:
  tag = intent['tag']
  tags.append(tag)

  for pattern in intent['patterns']:
    w = tokenize(pattern)
    all_words.extend(w) # as 'w' is an array, append will put an array in an array and extend will just add the elements in the other array.
    xy.append((w, tag)) # this will know the patterns and the corresponding tags, i.e, independent variables X and label Y respectively.

ignore_words = ['?', '!', ',', '.']
all_words = [stem(w) for w in all_words if w not in ignore_words]

all_words = sorted(set(all_words)) # remove duplicated tokens (words)
tags = sorted(set(tags))


# Creating bag of words
X_train = []
Y_train = []

for (pattern_sentence, tag) in xy:
  bag = bag_of_words(pattern_sentence, all_words)
  X_train.append(bag)

  label = tags.index(tag)
  Y_train.append(label) # Not considering 1 hot encoder, but rather CrossEntropyLoss

X_train = np.array(X_train)
Y_train = np.array(Y_train)

class ChatDataset(Dataset):
  def __init__(self):
    self.n_samples = len(X_train)
    self.x_data = X_train
    self.y_data = Y_train

  def __getitem__(self, index):
    return self.x_data[index], self.y_data[index]

  def __len__(self):
    return self.n_samples

# Hyper parameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 3000

dataset = ChatDataset()
# The ML alogorith is basically a batch learnig as there is no online learning possibility
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
  for (words, labels) in train_loader:
    words = words.to(device)
    labels = labels.to(device)

    # Forward
    outputs = model(words)
    loss = criterion(outputs, labels)

    # Backward and optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  if (epoch + 1) % 100 == 0:
    print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}') # .4f is used to print until 4 decimal values


print(f'final loss, loss={loss.item():.4f}')

# save the data

data = {
  "model_state": model.state_dict(),
  "input_size": input_size,
  "output_size": output_size,
  "hidden_size": hidden_size,
  "all_words": all_words,
  "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training is completed and file is saved to {FILE}')