from .train_image_process import *
from .train_labels_process import *
import numpy as np
from utility.tool import *
import os

split_factor = 0.6


class KaggleDataProvider(object):
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
        self.calculate_clip_size()
        self.get_train_val_data()

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
        i = 0 # for test, then only use one image
        for img_id, label in self.dict_imageID_label.items():
            if i == 10:
                break
            i += 1
            # first padding
            label_array = np.array(label)

            label_pad = np.pad(label_array, ((0, self.pad_h), (0, self.pad_w)), 'constant', constant_values=(0, 0))
            label = label_pad.tolist()
            image = self.dict_imageID_image[img_id]
            image_array = np.array(image)
            image_array = np.array(normalization(image_array), dtype=float)
            image_pad = np.pad(image_array, ((0, self.pad_h), (0, self.pad_w)), 'constant', constant_values=(0, 0))
            image = image_pad.tolist()

            # clip image and mask
            for r in range(self.row):  # first dimension
                start_r = r * self.img_size
                end_r = start_r + self.img_size
                for col in range(self.column):  # second dimension
                    start_col = col * self.img_size
                    end_col = start_col + self.img_size
                    label = np.array(label)
                    image = np.array(image)

                    mask_clip = label[start_r:end_r, start_col:end_col]
                    image_clip = image[start_r:end_r, start_col:end_col]
                    mask_clip = mask_clip.tolist()
                    image_clip = image_clip.tolist()
                    # print(np.array(mask_clip).shape)
                    # print(np.array(image_clip).shape)
                    # print(np.array(label).shape)
                    # print(np.array(image).shape)
                    self.labels.append(mask_clip)
                    self.images.append(image_clip)

        length_data = len(self.labels)
        train_length = int(split_factor * length_data)
        self.train_images = self.images[:train_length]
        self.train_labels = self.labels[:train_length]
        self.validate_images = self.images[train_length:]
        self.validate_labels = self.labels[train_length:]
