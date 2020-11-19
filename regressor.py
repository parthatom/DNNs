import torch
from torch import nn, optim
from torch.nn import functional as F

class dnn(nn.Module):
  def __init__(self, x_size = 2, hidden_list = [8,8,8], num_classes = 2, **kwargs):
    super(dnn, self).__init__()
    self.hidden_list = hidden_list
    self.x_size = x_size
    self.kwargs = kwargs
    self.num_classes = num_classes
    if ("batch_norm" in self.kwargs):
      self.batch_norm = self.kwargs['batch_norm']
    else:
      self.batch_norm = False
    if ("dropout_rate" in self.kwargs):
      self.dropout_rate = self.kwargs['dropout_rate']
    else:
      self.dropout_rate = 0.0

    dnn_list = []
    prev = self.x_size
    for hid in hidden_list:
      dnn_list.append(nn.Linear(prev,hid))
      if (self.batch_norm):
        dnn_list.append(nn.BatchNorm1d(hid))
      dnn_list.append(nn.ReLU())
      dnn_list.append( nn.Dropout(p=self.dropout_rate) )
      prev = hid

    dnn_list.append(nn.Linear(prev,1))
    if (self.batch_norm):
      dnn_list.append(nn.BatchNorm1d(1))
    # dnn_list.append(nn.LogSoftmax(dim= 1))
    dnn_list.append(nn.Tanh())

    self.net = nn.Sequential(*dnn_list)
  def forward(self,x):
    h = self.net(x)
    return h
