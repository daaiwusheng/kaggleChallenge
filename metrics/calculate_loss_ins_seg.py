import torch
from config.config import *

from metrics.loss import *

#  定义key值，避免无意义的key，
WEIGHT_MASK_KEY = 'weight_mask'
WEIGHT_MASK_ASSISTANT_KEY = 'weight_mask_assistant'
WEIGHT_CONTOUR_KEY = 'weight_contour'
WEIGHT_CONTOUR_ASSISTANT_KEY = 'weight_contour_assistant'
WEIGHT_DISTANCE_KEY = 'weight_distance'

dict_loss_weights = {
    WEIGHT_MASK_KEY: 1.0,
    WEIGHT_MASK_ASSISTANT_KEY: 0.5,
    WEIGHT_CONTOUR_KEY: 1.0,
    WEIGHT_CONTOUR_ASSISTANT_KEY: 0.5,
    WEIGHT_DISTANCE_KEY: 5.0
}


# 如果想改变权重的值，使用下面这个函数
def set_loss_weights(key: str, value: float):
    if key in dict_loss_weights.keys():
        dict_loss_weights[key] = value
        return True
    else:
        return False


class LossCalculator(nn.Module):
    def __init__(self, dict_weights=dict_loss_weights, split_channels:list=[1,1,1]):
        super(LossCalculator, self).__init__()
        self.split_channels = split_channels
        self.dict_weights = dict_weights
        self.mask_loss = WeightedBCEWithLogitsLoss()
        self.mask_loss_assis = DiceLoss()
        self.contour_loss = WeightedBCEWithLogitsLoss()
        self.contour_loss_assis = DiceLoss()
        self.distance_loss = WeightedMSE()

    def splitor(self, x):
        x = torch.split(x, self.split_channels, dim=1)
        x = list(x)  # torch.split returns a tuple
        return x

    def forward(self, predictions, mask_y, contour_y, distance_y):
        predictions_splited = self.splitor(predictions)

        distance_loss = self.dict_weights[WEIGHT_DISTANCE_KEY] * self.distance_loss(torch.tanh(predictions_splited[2]), distance_y)

        loss_result = self.dict_weights[WEIGHT_MASK_KEY] * self.mask_loss(predictions_splited[0], mask_y) \
                      + self.dict_weights[WEIGHT_MASK_ASSISTANT_KEY] * self.mask_loss_assis(torch.sigmoid(predictions_splited[0]), mask_y) \
                      + self.dict_weights[WEIGHT_CONTOUR_KEY] * self.contour_loss(predictions_splited[1], contour_y) \
                      + self.dict_weights[WEIGHT_CONTOUR_ASSISTANT_KEY] * self.contour_loss_assis(torch.sigmoid(predictions_splited[1]), contour_y) \
                      + distance_loss
        return loss_result, distance_loss
