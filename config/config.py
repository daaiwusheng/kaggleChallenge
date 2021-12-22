import torch
import numpy as np
#定义全局的数据类型，用于模型计算，和 loss计算，避免到处写，然后不好改

DATA_TYPE_TENSOR = torch.float
DATA_TYPE_NP = np.float



# TRAIN_CSV = "F:/LeedsDocs/Kaggle/sartorius-cell-instance-segmentation/train.csv"
# TRAIN_PATH = "F:/LeedsDocs/Kaggle/sartorius-cell-instance-segmentation/train/"
# MASK_DIR = "F:/LeedsDocs/Kaggle/sartorius-cell-instance-segmentation/train_mask.csv"
# TRAIN_PATCH_IMAGE_SAVE = "F:/LeedsDocs/Kaggle/sartorius-cell-instance-segmentation/train_patch_image"
# TRAIN_PATCH_MSAK_SAVE = "F:/LeedsDocs/Kaggle/sartorius-cell-instance-segmentation/train_patch_mask"
# LOGER_PATH = 'F:/LeedsDocs/Kaggle/exp.log' # On Tianyou's pc

TRAIN_CSV = "/nobackup/ml20t2w/Data/sartorius-cell-instance-segmentation/train.csv"
TRAIN_PATH = "/nobackup/ml20t2w/Data/sartorius-cell-instance-segmentation/train/"
MASK_DIR = "/nobackup/ml20t2w/Data/sartorius-cell-instance-segmentation/train_mask.csv"
TRAIN_PATCH_IMAGE_SAVE = "/nobackup/ml20t2w/Data/sartorius-cell-instance-segmentation/train_patch_image"
TRAIN_PATCH_MSAK_SAVE = "/nobackup/ml20t2w/Data/sartorius-cell-instance-segmentation/train_patch_mask"
LOGER_PATH = '/nobackup/ml20t2w/code/trains/log/exp.log' # On HPC


# TRAIN_CSV = "/content/drive/MyDrive/Project/sartorius-cell-instance-segmentation/train.csv"
# TRAIN_PATH = "/content/drive/MyDrive/Project/sartorius-cell-instance-segmentation/train/"
# MASK_DIR = "/content/drive/MyDrive/Project/sartorius-cell-instance-segmentation/train_mask.csv"
# RESULT_DIR = "/content/drive/MyDrive/Project/kaggle_result/"  # On Colab

# MASK_DIR = "/Users/wangyu/Desktop/利兹上课资料/Kaggle比赛/data/train_mask.csv"  # Steven Mac
# TRAIN_CSV = "/Users/wangyu/Desktop/利兹上课资料/Kaggle比赛/data/train.csv"  # Steven Mac
# TRAIN_PATH = "/Users/wangyu/Desktop/利兹上课资料/Kaggle比赛/data/train/"  # Steven Mac
# TRAIN_PATCH_IMAGE_SAVE = "/Users/wangyu/Desktop/利兹上课资料/Kaggle比赛/data/train_patch_image"  # Steven Mac
# TRAIN_PATCH_MSAK_SAVE = "/Users/wangyu/Desktop/利兹上课资料/Kaggle比赛/data/train_patch_mask"  # Steven Mac
# LOGER_PATH = 'F:/LeedsDocs/Kaggle/exp.log'

# MASK_DIR = "/home/steven/桌面/kaggle/data/train_mask.csv"  # Steven Linux
# TRAIN_CSV = "/home/steven/桌面/kaggle/data/train.csv"  # Steven Linux
# TRAIN_PATH = "/home/steven/桌面/kaggle/data/train/"  # Steven Linux
# TRAIN_PATCH_IMAGE_SAVE = "/databig/kaggleData/train_patch_image"  # Steven Linux
# TRAIN_PATCH_MSAK_SAVE = "/databig/kaggleData/train_patch_mask"  # Steven Linux
# LOGER_PATH = '/databig/kaggleLog/exp.log'  # Steven Linux


# TRAIN_CSV = "D:/Seg-cell/train.csv" # guanhui
# TRAIN_PATH = "D:/Seg-cell/train" # guanhui
# RESULT_DIR = "D:/Seg-cell" # guanhui
# MASK_DIR = "D:/Seg-cell/train_mask.csv" # guanhui

# TRAIN_CSV = "D:/Seg-cell/train.csv" # guanhui
# TRAIN_PATH = "D:/Seg-cell/train" # guanhui
# RESULT_DIR = "D:/Seg-cell" # guanhui
# MASK_DIR = "D:/Seg-cell/train_mask.csv" # guanhui


WIDTH = 704
HEIGHT = 520

BATCH_SIZE = 8

PATCH_SIZE = 64
