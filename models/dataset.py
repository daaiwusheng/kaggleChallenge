import torch
import numpy as np
from config.config import *
from torch.utils.data import Dataset
import collections
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from torchvision import transforms
from utility.rle_tool import *
from dataprocess.kaggle_data_provider import *
from torchvision import transforms as T
from typing import Callable
from scipy.ndimage import distance_transform_edt


## 输出的shape  img和masks:(1,520,704)  c,h,w


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class CellDataset(Dataset):
    def __init__(self, image_dir, df, split='train', transforms=None, resize=False, patch_size=16):
        self.transforms = transforms
        self.image_dir = image_dir
        self.df = df
        self.patch_size = patch_size
        self.split = split

        self.should_resize = resize is not False
        if self.should_resize:
            self.height = int(HEIGHT * resize)
            self.width = int(WIDTH * resize)
        else:
            self.height = HEIGHT
            self.width = WIDTH

        self.row_sum = self.height // self.patch_size
        self.col_sum = self.width // self.patch_size
        self.patch_num = self.row_sum * self.col_sum

        self.image_info = collections.defaultdict(dict)
        ##这一步将相同ID的annotation组成在一起。比如原文件关于id=001的annotation有400条(行），操作过后temp_df中id=001的len(annotation)=400（一行）.
        temp_df_11 = self.df.groupby('id')['annotation'].agg(lambda x: list(x)).reset_index()

        train_df, val_df = train_test_split(temp_df_11, test_size=0.2, random_state=42)
        if self.split == 'train':
            temp_df = train_df
        elif self.split == 'val':
            temp_df = val_df
        print(len(temp_df))
        ## 将id, path, annotation转成字典
        ii = 0
        for index, row in temp_df.iterrows():
            self.image_info[ii] = {
                'image_id': row['id'],
                'image_path': os.path.join(self.image_dir, row['id'] + '.png'),
                'annotations': row["annotation"]
            }
            ii += 1

    def __getitem__(self, idx):
        ''' Get the image and the target'''
        img_idx = idx // self.patch_num
        patch_idx = idx % self.patch_num

        img_path = self.image_info[img_idx]["image_path"]
        img = Image.open(img_path)

        if self.should_resize:
            img = img.resize((self.width, self.height), resample=Image.BILINEAR)

        info = self.image_info[img_idx]

        n_objects = len(info['annotations'])
        # ********************************************************************************#
        ##这一步得到的masks是将每一个a_mask放在不同的通道上，所有有多少a_mask就有多少通道
        ## boxes记录着每个a_mask的边框的顶点
        # masks = np.zeros((len(info['annotations']), self.height, self.width), dtype=np.uint8)
        # boxes = []

        # for i, annotation in enumerate(info['annotations']):
        #     a_mask = rle_decode(annotation, (HEIGHT, WIDTH))
        #     a_mask = Image.fromarray(a_mask)

        #     if self.should_resize:
        #         a_mask = a_mask.resize((self.width, self.height), resample=Image.BILINEAR)

        #     a_mask = np.array(a_mask) > 0
        #     masks[i, :, :] = a_mask

        #     boxes.append(self.get_box(a_mask))
        # ********************************************************************************#

        # ********************************************************************************#
        ##  这个写法可以让所有a_mask加在同一个通道上，看起来更正常点
        masks = np.zeros((self.height, self.width), dtype=np.uint8)

        for i, annotation in enumerate(info['annotations']):
            a_mask = rle_decode(annotation, (HEIGHT, WIDTH))
            a_mask = Image.fromarray(a_mask)

            if self.should_resize:
                a_mask = a_mask.resize((self.width, self.height), resample=Image.BILINEAR)

            a_mask = np.array(a_mask) > 0
            masks += a_mask

        # ********************************************************************************#

        masks = np.where(masks > 0, 1, 0)  # 大于0的地方取0，否则取1. 因为前面的a_mask在一些像素上重叠了，所以需要改成1
        img = np.array(img)

        # 计算patch位置
        cur_row = patch_idx // self.col_sum
        cur_col = patch_idx % self.col_sum

        start_row = cur_row * self.patch_size
        end_row = start_row + self.patch_size
        start_col = cur_col * self.patch_size
        end_col = start_col + self.patch_size
        # 切片
        mask_clip = masks[start_row:end_row, start_col:end_col]
        image_clip = img[start_row:end_row, start_col:end_col]

        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image_clip = transform(image_clip)
        mask_clip = transform(mask_clip)

        mask_clip = torch.as_tensor(mask_clip, dtype=torch.uint8)
        image_clip = torch.as_tensor(image_clip, dtype=torch.float)

        # if self.transforms is not None:
        #     img, masks = self.transforms(img, masks)

        return image_clip, mask_clip

    def __len__(self):
        return len(self.image_info) * (520 // self.patch_size) * (704 // self.patch_size)


class KaggleData(Dataset):
    def __init__(self, is_train=True, dataset_path: str = None, joint_transform: Callable = None, image_size=64):
        self.data_provider = KaggleDataProvider(image_size=image_size)
        self.images = []
        self.masks = []
        if is_train:
            self.images = self.data_provider.train_images
            self.masks = self.data_provider.train_labels
        else:
            self.images = self.data_provider.validate_images
            self.masks = self.data_provider.validate_labels

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        image = np.array(self.images[idx])
        mask = np.array(self.masks[idx])
        image, mask = correct_dims(image, mask)
        # print(image.shape)
        mask = np.where(mask > 0, 1, 0)  # 大于0的地方取0，否则取1. 因为前面的a_mask在一些像素上重叠了，所以需要改成1
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image = transform(image)
        mask = transform(mask)
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        image = torch.as_tensor(image, dtype=torch.float)
        # if self.joint_transform:
        #     image, mask = self.joint_transform(image, mask)

        return image, mask


class KaggleDatasetFromPatchFiles(Dataset):
    # 这个类从已经切好的patch 和 mask patch 读取文件,不会一下全部把图片读到内存中
    def __init__(self, is_train=True, dataset_path: str = None, joint_transform: Callable = None, image_size=64):
        self.data_provider = KaggleDataProviderFromPreparedFiles(image_size=image_size)
        self.images = []
        self.masks = []
        if is_train:
            self.images = self.data_provider.train_images
            self.masks = self.data_provider.train_labels
        else:
            self.images = self.data_provider.validate_images
            self.masks = self.data_provider.validate_labels
        self.joint_transform = transforms.Compose([
            transforms.ToTensor()
        ])


    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        mask_name = self.masks[idx]

        image = Image.open(image_name).convert("L")
        image_array = np.asarray(image, dtype=DATA_TYPE_NP)
        image_array = normalization(image_array)

        mask = Image.open(mask_name).convert("L")
        mask_array = np.asarray(mask, dtype='uint8')

        mask_array = np.where(mask_array > 0, 1, 0)  # 大于0的地方取0，否则取1. 因为前面的a_mask在一些像素上重叠了，所以需要改成1
        mask_binary_array = np.array(mask_array, dtype='uint8')  # 这里一定要用np.array， 因为这里需要copy

        image_array, mask_array = correct_dims(image_array, mask_array)
        # print(image.shape)

        # 从mask 中获取 contour
        binary_contuor_map = get_contour_from_mask(mask_binary_array)
        # 提取distance map

        distance_map = get_distance_edt(mask_binary_array)
        distance_map = np.tanh(distance_map)

        # convert numpy to tensor
        image_tensor = self.joint_transform(image_array)
        mask_tensor = self.joint_transform(mask_array)
        binary_contuor_map_tensor = self.joint_transform(binary_contuor_map)
        distance_map_tensor = self.joint_transform(distance_map)


        # mask_tensor = torch.as_tensor(mask_tensor, dtype=torch.uint8)
        # image_tensor = torch.as_tensor(image_tensor, dtype=torch.float)

        return image_tensor.float(), mask_tensor.float(), \
               binary_contuor_map_tensor.float(), distance_map_tensor.float()
