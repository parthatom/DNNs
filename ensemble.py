import torch
from torch import nn, optim
from torch.nn import functional as F
from DNNs import classifier

class voting_classifier(nn.Module):
    def __init__(self, m1_list, transformer_list, hard_voting = False):
        super(voting_classifier, self).__init__()
        self.m1_list = []
        self.transformer_list = []

        self._m1_list_check(m1_list)
        self._transformer_list_check(transformer_list)

        self.hard_voting = hard_voting

    def _m1_list_check(self, m1_list):
        for m in m1_list:
            if (isinstance(m, nn.Module)):
                self.m1_list.append(m)
            elif (hasattr(m, 'predict_proba')):
                self.m1_list.append(m.predict_proba)
            else:
                raise TypeError("Only torch/scikit learn models are supported")

    def _transformer_list_check(self, transformer_list):
        if (isinstance(transformer_list, list)):
            if (len (transformer_list) == len(self.m1_list)):
                self.transformer_list = transformer_list
            else:
                raise ValueError("Lenght of lists dont match")
        elif transformer_list is None:
            self.transformer_list = [None] * len(self.m1_list)
        else:
            raise TypeError("transformer_list must be a list of transformers or None")

    def forward(self, x):
        a = []
        for i,m in enumerate(self.m1_list):
          t = self.transformer_list[i]

          if (t is not None):
            b =  m (t(x))
          else:
            b = m(x)
          if (self.hard_voting):
            b = (b>0.5).double()

          if (len(a) > 0):
            a += b
          else:
            a = b
        a /= len(self.m1_list)
        return a

class stacking_classifier_dnn(nn.Module):
    def __init__(self, m1_list, transformer_list = None, num_classes = 2, hidden_list = []):
        """
        Stacking classifier
        ---
        Where the final classifier is a trainable DNN
        """
        super(stacking_classifier_dnn, self).__init__()
        self.m1_list = []
        self.transformer_list = []

        self._m1_list_check(m1_list)
        self._transformer_list_check(transformer_list)

        self.num_classes = num_classes
        self.x_size = self.num_classes * len(self.m1_list)
        self.hidden_list = hidden_list
        self.net = classifier.dnn(x_size = self.x_size, hidden_list = self.hidden_list, num_classes = self.num_classes)

    def _m1_list_check(self, m1_list):
        for m in m1_list:
            if (isinstance(m, nn.Module)):
                self.m1_list.append(m)
            elif (hasattr(m, 'predict_proba')):
                self.m1_list.append(m.predict_proba)
            else:
                raise TypeError("Only torch/scikit learn models are supported")

    def _transformer_list_check(self, transformer_list):
        if (isinstance(transformer_list, list)):
            if (len (transformer_list) == len(self.m1_list)):
                self.transformer_list = transformer_list
            else:
                raise ValueError("Length of lists dont match")
        elif transformer_list is None:
            self.transformer_list = [None] * len(self.m1_list)
        else:
            raise TypeError("transformer_list must be a list of transformers or None")

    def forward(self, x):
        preds = []
        for t,m in zip(self.m1_list, self.transformer_list):

            if (t is not None):
                _pred = m( t(x) )
            else:
                _pred = m(x)

            preds.append(_pred)
        preds = torch.cat(preds, dim = 1)

        return self.net(preds)
