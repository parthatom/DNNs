import torch
from torch import nn, optim
from torch.nn import functional as F

def dnn_accuracy(m1, transform, test1_loader, combine_x_c = True, printing = True):

  total = 0
  correct = 0

  for data in test1_loader:
    x = data[0]
    if (combine_x_c == True):
      x = torch.cat([x, data[1]], axis = 1)
    if (transform is not None):
      x = transform(x)
    y = data[2]
    y_pred = m1(x)
    _, predicted = torch.max(y_pred.data, 1)
    total += y.size(0)
    correct += (predicted == y).sum().item()
  if printing:
    print(correct/total)
  return correct/total

def get_preds(m1, test_loader, transform = None, x_index = None):
  preds = []
  with torch.no_grad():
    for data in test_loader:
      if x_index is not None:
        x = data[x_index]
      else:
        x=  data
      if transform is not None:
        x= transform(x)
      y_pred = m1(x)
      _, predicted = torch.max(y_pred.data, 1)
      preds.append(predicted)
  preds = torch.cat(preds, dim = 0)
  return preds
