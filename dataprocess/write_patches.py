from .train_image_process import *
from .train_labels_process import *

from PIL import Image
import numpy as np
from utility.tool import *
import os

split_factor = 0.6


class KaggleDataSaver(object):
    def __init__(self, image_size=64):
        self.images = []
        self.labels = []
        self.train_images = []
        self.train_labels = []
        self.validate_images = []
        self.validate_labels = []

        self.img_size = image_size
        self.image_provider = TrainImageHandler()
        self.label_provider = TrainLabelsProcessor()

        self.dict_imageID_image = self.image_provider.dict_imgID_image
        self.dict_imageID_label = self.label_provider.dict_imgID_mask

        self.pad_w = 0
        self.pad_h = 0
        self.column = 1
        self.row = 1

        self.dir_save_patches = TRAIN_PATCH_IMAGE_SAVE + '_' + str(self.img_size)
        dir_save_patches_exist_bool = os.path.exists(self.dir_save_patches)
        self.dir_save_maskes = TRAIN_PATCH_MSAK_SAVE + '_' + str(self.img_size)
        dir_save_maskes_exist_bool = os.path.exists(self.dir_save_maskes)
        valid_files_in_bool = self.test_patches_and_masks_in_dir()
        if dir_save_patches_exist_bool and dir_save_maskes_exist_bool and valid_files_in_bool:
            #  get patches and masks path
            self.get_patches_masks_fullpath()
            self.split_train_val_data()

        else:

            if dir_save_patches_exist_bool:
                os.rmdir(self.dir_save_patches)
            if dir_save_maskes_exist_bool:
                os.rmdir(self.dir_save_maskes)
            os.mkdir(self.dir_save_patches)
            os.mkdir(self.dir_save_maskes)
            self.calculate_clip_size()
            self.get_train_val_data()
            self.get_patches_masks_fullpath()
            self.split_train_val_data()

    def calculate_clip_size(self):
        image_width = 704
        image_height = 520

        threshold = round(0.5 * self.img_size)
        m_w = image_width % self.img_size
        m_h = image_height % self.img_size

        if m_w < threshold:
            self.column = image_width // self.img_size
        else:
            self.column = image_width // self.img_size + 1
            self.pad_w = self.img_size - m_w

        if m_h < threshold:
            self.row = image_height // self.img_size
        else:
            self.row = image_height // self.img_size + 1
            self.pad_h = self.img_size - m_h

    def get_train_val_data(self):
        max_test_number = 0  # for test, then only use one image
        for img_id, label in self.dict_imageID_label.items():
            if max_test_number == 1:
                break
            max_test_number += 1
            # first padding
            label_array = np.array(label)

            label_pad = np.pad(label_array, ((0, self.pad_h), (0, self.pad_w)), 'constant', constant_values=(0, 0))
            label = label_pad.tolist()
            image = self.dict_imageID_image[img_id]
            image_array = np.array(image)
            #   #  先不要归一化,不然保存图片全是黑色的,不知道为什么
            # image_array = np.array(normalization(image_array), dtype=float)
            image_pad = np.pad(image_array, ((0, self.pad_h), (0, self.pad_w)), 'constant', constant_values=(0, 0))
            image = image_pad.tolist()

            # clip image and mask
            current_idx = 0
            for r in range(self.row):  # first dimension
                start_r = r * self.img_size
                end_r = start_r + self.img_size
                for col in range(self.column):  # second dimension
                    start_col = col * self.img_size
                    end_col = start_col + self.img_size

                    label = np.array(label, dtype=float)
                    image = np.array(image, dtype=float)

                    image_clip_data = image[start_r:end_r, start_col:end_col]
                    mask_clip_data = label[start_r:end_r, start_col:end_col]

                    current_idx = r * self.column + col
                    image_clip = Image.fromarray(image_clip_data)
                    mask_clip = Image.fromarray(mask_clip_data)

                    image_clip_name = self.dir_save_patches + '/' + img_id + '_' + str(
                        current_idx) + '_' + 'image' + '.png'
                    mask_clip_name = self.dir_save_maskes + '/' + img_id + '_' + str(
                        current_idx) + '_' + 'mask' + '.png'

                    image_clip = image_clip.convert("L")
                    mask_clip = mask_clip.convert("L")

                    image_clip.save(image_clip_name)
                    mask_clip.save(mask_clip_name)

    def split_train_val_data(self):
        length_data = len(self.labels)
        train_length = int(split_factor * length_data)
        self.train_images = self.images[:train_length]
        self.train_labels = self.labels[:train_length]
        self.validate_images = self.images[train_length:]
        self.validate_labels = self.labels[train_length:]

    def test_patches_and_masks_in_dir(self):
        imgs_in_bool = False
        masks_in_bool = False
        for root, dirs, files in os.walk(self.dir_save_patches, topdown=True):
            for image_name in files:
                if '.png' in image_name:
                    imgs_in_bool = True

        for root, dirs, files in os.walk(self.dir_save_maskes, topdown=True):
            for mask_clip_name in files:
                if '.png' in mask_clip_name:
                    masks_in_bool = True
        return imgs_in_bool and masks_in_bool

    def get_patches_masks_fullpath(self):
        for root, dirs, files in os.walk(self.dir_save_patches, topdown=True):
            files.sort()
            for image_name in files:
                if '.png' in image_name:
                    image_full_path = os.path.join(self.dir_save_patches, image_name)
                    self.images.append(image_full_path)

        for root, dirs, files in os.walk(self.dir_save_maskes, topdown=True):
            files.sort()
            for mask_clip_name in files:
                if '.png' in mask_clip_name:
                    image_full_path = os.path.join(self.dir_save_maskes, mask_clip_name)
                    self.labels.append(image_full_path)
