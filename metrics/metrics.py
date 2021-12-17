
import torch
import torch.nn as nn
import torch.nn.functional as F


class IoUScore(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoUScore, self).__init__()

    def forward(self, inputs, targets, smooth=1):



        # inputs = F.softmax(inputs, dim=1)[:, 1:]



        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)


        return IoU.cpu().numpy()

class InstanceIoUScore(nn.Module):
    def __init__(self, weight=None, split_channels:list=[1,1,1]):
        super(InstanceIoUScore, self).__init__()
        self.weight = weight
        self.split_channels = split_channels
        self.mask_iou = IoUScore()
        self.contour_iou = IoUScore()
        self.distance_iou = IoUScore()

    def splitor(self, x):
        x = torch.split(x, self.split_channels, dim=1)
        x = list(x)  # torch.split returns a tuple
        return x

    def forward(self, inputs, mask_tensor, binary_contuor_map_tensor, distance_map_tensor):
        inputs_splited = self.splitor(inputs)

        iou = self.mask_iou(torch.sigmoid(inputs_splited[0]), mask_tensor) \
              + self.contour_iou(torch.sigmoid(inputs_splited[1]), binary_contuor_map_tensor) \
              + self.distance_iou(torch.tanh(inputs_splited[2]), distance_map_tensor)

        return iou



thres = 0.5

def compute_metrics(skeleton_output, skeleton_gt, skeleton_output_dil, skeleton_gt_dil):

    """
    inputs:
    skeleton_output - list containing skeletonized network probability maps after binarization at 0.5
    skeleton_gt - list containing skeletonized groun-truth images
    skeleton_output_dil - list containing skeletonized outputs dilated by the factor N
    skeleton_gt_dil - list containing skeletonized ground truth images dilated by the factor N
    """

    tpcor = 0
    tpcom = 0
    fn = 0
    fp = 0

    for i in range(0, len(skeleton_output)):
        tpcor += ((skeleton_output[i] == skeleton_gt_dil[i]) & (skeleton_output[i] == 1)).sum()
        tpcom += ((skeleton_gt[i]==skeleton_output_dil[i]) & (skeleton_gt[i]==1)).sum()
        fn += (skeleton_gt[i]==1).sum() - ((skeleton_gt[i]==skeleton_output_dil[i]) & (skeleton_gt[i]==1)).sum()
        fp += (skeleton_output[i]==1).sum() - ((skeleton_output[i] == skeleton_gt_dil[i]) & (skeleton_output[i] == 1)).sum()

    correctness = tpcor/(tpcor+fp)
    completness = tpcom/(tpcom+fn)
    if (completness - completness*correctness + correctness) == 0.0:
        quality = 0.0
    else:
        quality = completness*correctness/(completness - completness*correctness + correctness)

    return correctness, completness, quality

def compute_precision_recall(pred, gt):
    pred_skel = skeletonize(pred)
    pred_dil = dilation(pred_skel, square(5))
    gt_skel = skeletonize(gt)
    gt_dil = dilation(gt_skel, square(5))
    return compute_metrics([pred_skel], [gt_skel], [pred_dil], [gt_dil])

def calc_iou(pred, gt):
    # Both the pred and gt are binarized.
    # Calculate foreground iou:
    inter = np.logical_and(pred, gt).astype(np.float32)
    union = np.logical_or(pred, gt).astype(np.float32)
    if union.sum() == 0:
        foreground_iou = 0.0
    else:
        foreground_iou = inter.sum() / union.sum()
    return foreground_iou

def binarize(pred, gt):
    pred = (pred > thres).astype(np.uint8)
    gt = (gt!=0).astype(np.uint8) * (gt!=255).astype(np.uint8)
    return pred, gt

def instancemetric(pred, gt):
    pred, gt = binarize(pred, gt)
    num_gt = gt.sum()
    if num_gt == 0:
        return 1.0, 1.0, 1.0, 1.0
    else:
        foreground_iou = calc_iou(pred, gt)
        correctness, completness, quality = compute_precision_recall(pred, gt)
        return foreground_iou, correctness, completness, quality