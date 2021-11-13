import torch
from torch.nn.modules.loss import _WeightedLoss
from torch.nn.functional import cross_entropy


class LogNLLLoss(_WeightedLoss):
    __constants__ = ['weight', 'reduction', 'ignore_index']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction=None,
                 ignore_index=-100):
        super(LogNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, y_input, y_target):
        # y_input = torch.log(y_input + EPSILON)

        y_input = torch.squeeze(y_input)
        y_target = torch.squeeze(y_target)
        y_target = torch.as_tensor(y_target, dtype=torch.long)

        print(y_input.shape)
        print(y_target.shape)
        return cross_entropy(y_input, y_target, weight=self.weight,
                             ignore_index=self.ignore_index)