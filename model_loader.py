import os

import numpy as np
import pandas as pd

import torch
from DNNs import classifier

class model_loading():
  def __init__(self, log_path, x_size = 37):
    self.x_size = int (x_size)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.log_path = log_path

  def load_dnn(self, row):

    assert (isinstance(row, pd.Series) or isinstance(row, dict))

    hidden_list = row['hidden_dims'].replace('[','').replace(']','')
    hidden_list = list (map(int, hidden_list.split(",")))
    if (os.path.exists(os.path.join(self.log_path,row['dir_name'], "model_optimizer_statae_dict.pt"))):
      checkpoint = torch.load(os.path.join(self.log_path,row['dir_name'], "model_optimizer_statae_dict.pt"), map_location=self.device)
    else:
      print ("failed to Load DNN, path doesn't exist")
      return None
    self.m1 = classifier.dnn(x_size = self.x_size,
                              hidden_list=hidden_list,
                              num_classes = int (row['num_classes']),
                              batch_norm = True, dropout_rate = 0.1,
                              last_batch_norm = row['last_batch_norm'] == "True")
    self.m1.load_state_dict(checkpoint['model_state_dict'])
    return self.m1
