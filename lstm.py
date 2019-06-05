from __future__ import division
import torch
from torch import nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib
import random
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

###
# Author: Clay O'Neil
# Implementation of Long Short-Term Memory
# https://www.bioinf.jku.at/publications/older/2604.pdf
###

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class LSTM(nn.Module):

  ### PARAMETERS 
  # dim is number of hidden units
  # max_len is the maximum length of the input
  # batch_size is length of minibatch
  # output_size is dim of final output
  # init_hidden (optional) is the initial hidden state
  ### RETURNS
  # outputs, size (batch_size, output_size)
  # hidden_states, size (batch_size, max_length, dim)
  ###

  def __init__(self, dim, max_len, batch_size, output_size, init_hidden=0, init_cell=0):
    super(LSTM, self).__init__()

    self.max_len = max_len
    self.dim = dim
    self.output_size = output_size
    self.batch_size = batch_size

    # Weight matrix for each of the 4 gates
    self.W = torch.randn(4 * dim, max_len + dim, device=device, dtype=torch.float)

    # Bias matrix
    self.b = torch.randn(4 * dim, device=device, dtype=torch.float)
    
    # Set forget bias to 1 per http://proceedings.mlr.press/v37/jozefowicz15.pdf
    self.b[:dim] = 1

    # initial hidden state
    self.h = torch.randn(dim, device=device, dtype=torch.float) if init_hidden == 0 else init_hidden

    # initial cell state
    self.c = torch.randn(dim, device=device, dtype=torch.float) if init_cell == 0 else init_cell

    # Projection layer
    self.projection = torch.nn.Linear(dim, output_size)

  # x is batch of input column vectors shape (batch_size, max_len)
  def forward(self, x):
    batch_hidden_states = []
    outputs = []

    for i in range(self.batch_size):
      hidden_states = []
      x_t = x[i]

      for j in range(self.max_len):
        concat_input = torch.cat((self.h, x_t[j]), 0).view(-1, 1)
        whx = self.W.mm(concat_input).view(-1) + self.b

        f_t = nn.Sigmoid() (whx[:self.dim])
        o_t = nn.Sigmoid() (whx[self.dim:self.dim * 2])
        i_t = nn.Sigmoid() (whx[self.dim * 2:self.dim * 3])
        ch_t = nn.Tanh() (whx[self.dim * 3:])

        self.c = f_t * self.c + i_t * ch_t
        self.h = o_t * (nn.Tanh() (self.c))
        
        hidden_states.append(self.h)
      logits = self.projection(self.h)
      output = torch.nn.Softmax()(logits)
      outputs.append(output)


      batch_hidden_states.append(hidden_states)

    return outputs, batch_hidden_states
      
def one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

# Cross Entropy Log Loss
def loss_fn(pred, gt):
  r = 0
  for i, row in enumerate(pred):
    r += torch.log(row.dot(gt[i].view(-1)) + 1e-8)
  return -1 * (r / len(pred))
      
lstm = LSTM(64, 10, 32, 10)

# Create dataset

x_data = []
y_data = []
for i in range(400):
  x = random.choices(range(10), k=10)
  y = [x[2]]
  x_data.append(one_hot(x, 10))
  y_data.append(one_hot(y, 10))

dataset = torch.utils.data.TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data))
#dataset = torch.utils.data.TensorDataset(
#              torch.Tensor([one_hot(list(range(lstm.max_len)), lstm.max_len) for _ in range(100)]), 
#              torch.Tensor([one_hot([9], lstm.max_len) for _ in range(100)]))
dataloader = DataLoader(dataset, batch_size=lstm.batch_size, shuffle=True, drop_last=True)

optimizer = torch.optim.Adam(lstm.parameters(), lr=3e-4)
epochs = 30
losses = []

for epoch in range(epochs):
  for batch_idx, batch in enumerate(dataloader):
    x = batch[0]
    y = batch[1]
    outputs, hidden_states = lstm(x)
    loss = loss_fn(outputs, y)
    print(batch_idx + epoch * lstm.batch_size, loss.item())
    losses.append(loss.item())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

t = np.linspace(0, 1000, len(losses))
y = np.cos(np.pi * (t / len(losses)))

plt.scatter(t, losses, c=y, s=1)

plt.axis([-1, 1000, -.01, 3])
plt.xlabel('batches', fontsize=14, color='red')
plt.ylabel('loss', fontsize=14, color='red')
plt.show()
