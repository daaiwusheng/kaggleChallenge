import torch
from torch.nn.modules.loss import _WeightedLoss
from torch.nn.functional import cross_entropy
import torch.nn as nn
import torch.nn.functional as F


def LogNLLLoss(y_input, y_target):
    # y_input = torch.log(y_input + EPSILON)
    y_input = torch.squeeze(y_input)
    y_target = torch.squeeze(y_target)
    y_target = torch.as_tensor(y_target, dtype=torch.long)

    # print(y_input.shape)
    # print(y_target.shape)
    return cross_entropy(y_input, y_target)


class DiceLoss(nn.Module):
    """DICE loss.
    """
    # https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/

    def __init__(self, size_average=True, reduce=True, smooth=100.0, power=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduce = reduce
        self.power = power

    def dice_loss(self, pred, target):
        loss = 0.

        for index in range(pred.size()[0]):
            iflat = pred[index].contiguous().view(-1)
            tflat = target[index].contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            if self.power == 1:
                loss += 1 - ((2. * intersection + self.smooth) /
                             (iflat.sum() + tflat.sum() + self.smooth))
            else:
                loss += 1 - ((2. * intersection + self.smooth) /
                             ((iflat**self.power).sum() + (tflat**self.power).sum() + self.smooth))

        # size_average=True for the dice loss
        return loss / float(pred.size()[0])

    def dice_loss_batch(self, pred, target):
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        if self.power == 1:
            loss = 1 - ((2. * intersection + self.smooth) /
                        (iflat.sum() + tflat.sum() + self.smooth))
        else:
            loss = 1 - ((2. * intersection + self.smooth) /
                        ((iflat**self.power).sum() + (tflat**self.power).sum() + self.smooth))
        return loss

    def forward(self, pred, target, weight_mask=None):
        if not (target.size() == pred.size()):
            raise ValueError("Target size ({}) must be the same as pred size ({})".format(
                target.size(), pred.size()))

        if self.reduce:
            loss = self.dice_loss(pred, target)
        else:
            loss = self.dice_loss_batch(pred, target)
        return loss


class WeightedMSE(nn.Module):
    """Weighted mean-squared error.
    """

    def __init__(self):
        super(WeightedMSE, self).__init__()

    def weighted_mse_loss(self, pred, target, weight=None):
        s1 = torch.prod(torch.tensor(pred.size()[2:]).float())
        s2 = pred.size()[0]
        norm_term = (s1 * s2).to(pred.device)
        if weight is None:
            return torch.sum((pred - target) ** 2) / norm_term
        return torch.sum(weight * (pred - target) ** 2) / norm_term

    def forward(self, pred, target, weight_mask=None):
        return self.weighted_mse_loss(pred, target, weight_mask)

class WeightedBCEWithLogitsLoss(nn.Module):
    """Weighted binary cross-entropy with logits.
    """

    def __init__(self, size_average=True, reduce=True):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, pred, target, weight_mask=None):
        return F.binary_cross_entropy_with_logits(pred, target, weight_mask)