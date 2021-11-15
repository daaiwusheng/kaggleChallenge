import torch
from torch.nn.modules.loss import _WeightedLoss
from torch.nn.functional import cross_entropy




def LogNLLLoss(y_input, y_target):
    # y_input = torch.log(y_input + EPSILON)
    y_input = torch.squeeze(y_input)
    y_target = torch.squeeze(y_target)
    y_target = torch.as_tensor(y_target, dtype=torch.long)

    # print(y_input.shape)
    # print(y_target.shape)
    return cross_entropy(y_input, y_target)