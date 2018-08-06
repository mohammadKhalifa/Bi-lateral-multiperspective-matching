import torch.nn.functional as F
from torch.nn import BCELoss

def my_loss(y_input, y_target):
    loss = BCELoss()
    return loss(y_input, y_target)
