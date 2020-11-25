import os
import numbers
import numpy as np
import pandas as pd

class Logger():
  def __init__(self, df_path, create = False, **kwargs):
    """
    Parameters:
    ==========
    df_path: path to csv file
    create: bool, if set to true, new DataFrame will be created,
            otherwise it is loaded from the path
    Other Parameters:
    ==========
    verbose: bool, if set to true, tells you about new columns added to DataFrame
    """
    self.kwargs = kwargs
    if ("verbose" in self.kwargs):
      self.verbose = self.kwargs['verbose']
    else:
      self.verbose = False
    self.df_path = df_path
    if (create or (not os.path.exists(self.df_path))):
      self.df = pd.DataFrame(columns = ['val_accuracy'])
    else:
      self.df = pd.read_csv(self.df_path, index_col=False)
    self.ind = len(self.df)
    self.df.loc[self.ind] = np.nan

  def log(self, key, value):
    """
    Logs the `value` for the corresponding `key`
    """
    if (self.verbose):
      print(f"{key}\t-- {value}")
    self.write_df(key, value, self.ind)

  def write_df(self, key, value, ind):
    if not key in self.df:
      if (self.verbose):
        print(key, "not in DataFrame, column has been added")
      self.df[key]=np.nan
      if (not isinstance(value, numbers.Number) ):
        self.df[key] = self.df[key].astype(str)
    self.df.at[ind, key] = value

  def next(self):
    """
    Next
    """
    self.ind += 1

  def save(self):
    """
    Saves the DataFrame at the provided datapath
    """
    self.df.to_csv(self.df_path, index = False)
