class voting_classifier(nn.Module):
  def __init__(self, m1_list, transformer_list, hard_voting = False):
    super(voting_classifier, self).__init__()
    self.m1_list = m1_list
    self.transformer_list = transformer_list
    self.hard_voting = hard_voting

  def forward(self, x):
    a = []
    for i,m in enumerate(self.m1_list):
      t = self.transformer_list[i]

      if (t is not None):
        b = F.softmax( m (t(x)), dim = 1)
      else:
        b = F.softmax( m(x), dim = 1)
      if (self.hard_voting):
        b = (b>0.5).double()

      if (len(a) > 0):
        a += b
      else:
        a = b
    a /= len(self.m1_list)
    return a
