# import pickle
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# criterion = nn.L1Loss()
#
# # dset = dataset(data_path, normalized = False, suff = 'h_q', vecs = True)
# # train_data,val_data = torch.utils.data.random_split(dset, lengths = [int (0.9*len(dset)),len(dset) - int (0.9*len(dset))])
# # dataloader = DataLoader(train_data, batch_size=1024,shuffle=True)
# # val_loader = DataLoader(val_data, batch_size=1024, shuffle=False)
#
# # lr = 0.0003
# lr_list = [0.0006,0.0004, 0.0003,0.00023,0.00022, 0.0002][::-1]
# for i in range(len(lr_list) - 2):
#   lr = torch.zeros((1)).uniform_(lr_list[i], lr_list[i+1]).item()
#   net = dnn(x_size = 768, hidden_list = [512, 256,128, 64, 32, 8], batch_norm = False, dropout_rate = 0.1)
#   net.to(device)
#   optimizer = optim.Adam(net.parameters(), lr=lr)
#
#
#
#   epochs = 30
#
#   start_time = time.time()
#   comments = ""
#   for keys in net.kwargs:
#     comments += str (keys) + "=" + str (net.kwargs[keys]) + ","
#
#   specifications = f'model=DNN,alpha={lr},epochs={epochs},batch_size={dataloader.batch_size},x_size={net.x_size},hidden={net.hidden_list},normalized={dataloader.dataset.dataset.normalized},{comments},{start_time}'
#   print(f'Training started for {specifications}')
#
#   log_path = os.path.join(data_path, specifications)
#   os.mkdir(log_path)
#
#   losses = {'loss':[], 'val_loss':[]}
#   # maes = {'mae' :[], 'val_mae' : []}
#   # accuracies = {'Accuracy':[], 'val_accuracy':[]}
#   min_val_loss = 10
#   for epoch in range(epochs):  # loop over the dataset multiple times
#
#       running_loss = 0.0
#       net.train()
#       correct = 0.0
#       total = 0.0
#       # mae = 0
#       for i, data in enumerate(dataloader, 0):
#           inputs, labels = data[0], data[1]
#           # c = data[1]
#           # inputs = torch.cat([inputs, c], dim =1)
#           # labels = labels.long()
#           labels = labels.reshape((len(labels), 1))
#           inputs = inputs.to(device)
#           labels = labels.to(device)
#           # zero the parameter gradients
#           optimizer.zero_grad()
#
#           # forward + backward + optimize
#           outputs = net(inputs.float())
#           loss = criterion(outputs, labels)
#           assert (net.training)
#           loss.backward()
#           optimizer.step()
#           # mae +=
#           # print statistics
#           running_loss += loss.item()
#           # if i % 2000 == 1999:    # print every 2000 mini-batches
#           #     print('[%d, %5d] loss: %.3f' %
#           #           (epoch + 1, i + 1, running_loss / 2000))
#           #       running_loss = 0.0
#       epoch_loss = running_loss/len(dataloader)
#       losses['loss'].append(epoch_loss)
#
#
#       # if (epoch%5==0):
#       print(f'Epoch:{epoch+1}/{epochs}, Training Loss: {epoch_loss:.5f}')
#         # print(f"Training Accuracy:{accuracy:.2f}%")
#
#       with torch.no_grad():
#         net.eval()
#
#         val_loss = 0.0
#         correct = 0.0
#         total = 0.0
#         for i, data in enumerate(val_loader, 0):
#           inputs, labels = data[0], data[1]
#           # c = data[1]
#           # inputs = torch.cat([inputs, c], dim =1)
#           # labels = labels.long()
#           labels = labels.reshape((len(labels), 1))
#           inputs = inputs.to(device)
#           labels = labels.to(device)
#
#           outputs = net(inputs.float())
#           loss = criterion(outputs, labels)
#
#           # if (epoch%5==0):
#
#           val_loss += loss.item()
#
#         val_loss = val_loss/len(val_loader)
#         min_val_loss = min(val_loss, min_val_loss)
#         if (val_loss == min_val_loss):
#           torch.save({'model_state_dict':net.state_dict(), 'optimizer_state_dict' : optimizer.state_dict()} , os.path.join(log_path, "model_optimizer_statae_dict.pt") )
#         losses['val_loss'].append(val_loss)
#
#         # if (epoch%5==0):
#         print(f'Validation Loss: {val_loss:.5f}')
#   import numbers
#   df_dnn = pd.read_csv(os.path.join(data_path, 'dnn_log.csv'))
#   ind = len(df_dnn)
#   def write_df(key, value, ind):
#     df = df_dnn
#     if not key in df:
#       print(key, "not in df, column has been added")
#       df[key]=np.nan
#       if (not isinstance(value, numbers.Number) ):
#         df[key] = df[key].astype(str)
#     df.at[ind, key] = value
#   write_df('loss', epoch_loss, ind)
#   write_df('val_loss', min_val_loss, ind)
#   write_df('epochs', epochs, ind)
#   write_df('alpha',lr, ind)
#   write_df('hidden_dims', net.hidden_list, ind)
#   write_df('normalized', int (dataloader.dataset.dataset.normalized), ind)
#   write_df("batch_size", dataloader.batch_size, ind)
#   write_df("comments", comments, ind)
#   write_df("start_time", start_time, ind)
#   write_df("optimizer", type(optimizer).__name__, ind)
#   write_df("dir_name", specifications, ind)
#   write_df("loss_func", type(criterion).__name__, ind)
#   write_df("suff", dataloader.dataset.dataset.suff, ind)
#   write_df("vecs", int (dataloader.dataset.dataset.vecs), ind)
#   df_dnn.to_csv(os.path.join(data_path, 'dnn_log.csv'), index=False)
#   with open(os.path.join(log_path, "losses.pkl"), 'wb') as f:
#     pickle.dump(losses, f)

import numbers
import os
import shutil
from pathlib import Path
import importlib
import time

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy
from scipy.io import loadmat

import sklearn
from sklearn import svm, metrics

import torch
from torch import nn, optim
from torch.nn import functional as F

import DNNs.Logging
from DNNs.Logging import Logger

class Trainer(object):
    """docstring for Trainer."""

    def __init__(self, model, optimizer, sizes, device, log_path,data_path,viz_freq=10, model_spec="classifier", writing = True, combine_x_c = False, **kwargs):
        super(Trainer, self).__init__()
        self.net = model
        self.optimizer = optimizer
        self.model_spec = model_spec
        assert (self.model_spec == "classifier" or self.model_spec == "regressor")
        self.z_size = sizes['z']
        self.c_size = sizes['c']
        self.x_size = sizes['x']
        self.device = device
        self.viz_freq=viz_freq
        self.data_path = data_path
        self.writing = writing
        self.combine_x_c = combine_x_c

        self.add_comments = self.net.kwargs
        self.num_batches_processed = 0
        self.epochs_processed = 0
        self.net.to(device)
        self.log_path = log_path
        self.acc_logger = None
        self.loss_logger = None
        self.kwargs = kwargs

        if ("early_stopping" in self.kwargs):
            self.early_stopping = self.kwargs['early_stopping']
        else:
            self.early_stopping = np.inf


    def train(self, epochs, dataloader, val_loader, transform = None, comments = ""):
        criterion = nn.NLLLoss()
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']

        features_used = "x" + str(self.x_size - (int(self.combine_x_c) * sum(self.c_size))) + int(self.combine_x_c) * "+context"
        start_time = time.time()
        for keys in self.net.kwargs:
          comments += str (keys) + "=" + str (self.net.kwargs[keys]) + ","

        specifications = f'model=DNN,alpha={lr:.4f},epochs={epochs},batch_size={dataloader.batch_size},x_size={self.net.x_size},hidden={",".join(list (map(str, self.net.hidden_list)))},normalized={dataloader.dataset.dataset.normalized},{comments},{start_time}'
        print(f'Training started for {specifications}')
        try:
            os.mkdir(os.path.join(self.log_path, specifications))
        except:
            specifications = f'model=DNN,x_size={self.net.x_size},hidden={",".join(list (map(str, self.net.hidden_list)))},{comments},{start_time}'
            print(f"Path too Long, renamed to {specifications}")
            os.mkdir(os.path.join(self.log_path, specifications))
        self.loss_logger = Logger.Logger(os.path.join(self.log_path, specifications, "losses.csv"), create=True, verbose = False)
        self.acc_logger = Logger.Logger(os.path.join(self.log_path, specifications, "accuracies.csv"), create=True, verbose = False)

        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            self.net.train()
            correct = 0.0
            total = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data[0], data[2]
                c = data[1]
                if (self.combine_x_c):
                    inputs = torch.cat([inputs, c], dim =1)
                if (transform is not None):
                  with torch.no_grad():
                    if (isinstance(transform, nn.Module)):
                        transform.to(self.device)
                        inputs = inputs.to(self.device)
                    inputs = transform (inputs)
                labels = labels.long()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                assert (self.net.training)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                running_loss += loss.item()

            epoch_loss = running_loss/len(dataloader)
            self.loss_logger.log('loss',epoch_loss)

            accuracy = 100*(correct/total)
            self.acc_logger.log('Accuracy',accuracy)

            if (epoch%5==0):
              print(f'Epoch:{epoch+1}/{epochs}, Training Loss: {epoch_loss:.2f}')
              print(f"Training Accuracy:{accuracy:.2f}%")

            with torch.no_grad():
              self.net.eval()

              val_loss = 0.0
              correct = 0.0
              total = 0.0
              for i, data in enumerate(val_loader, 0):
                inputs, labels = data[0], data[2]
                c = data[1]
                if (self.combine_x_c):
                    inputs = torch.cat([inputs, c], dim =1)
                if (transform is not None):
                  with torch.no_grad():
                    if (isinstance(transform, nn.Module)):
                        transform.to(self.device)
                        inputs = inputs.to(self.device)
                    inputs = transform (inputs)
                labels = labels.long()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                val_loss += loss.item()

              val_loss = val_loss/len(val_loader)
              self.loss_logger.log('val_loss',val_loss)

              val_accuracy = 100*(correct/total)
              if(epoch > self.early_stopping):
                  if (val_accuracy > max (self.acc_logger.df.val_accuracy)):
                      torch.save({'model_state_dict':self.net.state_dict(), 'optimizer_state_dict' : self.optimizer.state_dict()} , os.path.join(self.log_path, specifications, "model_optimizer_statae_dict.pt") )
              self.acc_logger.log("val_accuracy", val_accuracy)
              if (epoch%5==0):
                print(f'Validation Loss: {val_loss:.2f}')
                print(f"Validation Accuracy:{val_accuracy:.2f}%")

              self.loss_logger.next()
              self.acc_logger.next()

        self.loss_logger.save()
        self.acc_logger.save()
        if (self.early_stopping < epochs):
            final_accuracy =  max (self.acc_logger.df.Accuracy)
            final_val_accuracy = max(self.acc_logger.df.val_accuracy)
        else:
            final_accuracy = accuracy
            final_val_accuracy = val_accuracy

        print(f"Final Training Accuracy:{final_accuracy:.2f}%")
        print(f"Final Validation Accuracy:{final_val_accuracy:.2f}%")
        if (self.early_stopping > epochs):
            torch.save({'model_state_dict':self.net.state_dict(), 'optimizer_state_dict' : self.optimizer.state_dict()} , os.path.join(self.log_path, specifications, "model_optimizer_statae_dict.pt") )

        self.logger = Logger.Logger(os.path.join(self.data_path, "dnn_log.csv"), create=False, verbose = False)
        self.logger.log('loss', epoch_loss)
        self.logger.log('val_loss', val_loss)
        self.logger.log('Accuracy', final_accuracy)
        self.logger.log('val_accuracy', final_val_accuracy)
        self.logger.log('epochs', epochs)
        self.logger.log('alpha',lr)
        self.logger.log('hidden_dims', self.net.hidden_list)
        self.logger.log('normalized', float (dataloader.dataset.dataset.normalized))
        self.logger.log("batch_size", dataloader.batch_size)
        self.logger.log("comments", comments)
        self.logger.log("start_time", start_time)
        self.logger.log("optimizer", type(self.optimizer).__name__)
        self.logger.log("dir_name", specifications)
        self.logger.log("loss_func", type(criterion).__name__)
        self.logger.log("features_used", features_used)
        self.logger.log("early_stopping", self.early_stopping)
        self.logger.save()
