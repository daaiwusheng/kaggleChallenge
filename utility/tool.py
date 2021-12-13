import sys
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

csv.field_size_limit(sys.maxsize)


def outclude_hidden_files(files):
    return [f for f in files if not f[0] == '.']


def outclude_hidden_dirs(dirs):
    return [d for d in dirs if not d[0] == '.']


def show_image(image, window_name='test'):
    if isinstance(image, torch.Tensor):
        image = image.numpy()

    image = image.astype(np.uint8)
    image[image == 1] = 255
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_gray_image(image_array, vmin=0, vmax=1):
    plt.imshow(image_array, cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()


# encoding=utf-8
def get_cur_info():
    file_name = sys._getframe().f_code.co_filename  # 当前文件名，可以通过__file__获得
    function_name = sys._getframe().f_code.co_name  # 当前函数名
    line_num = get_line_num()
    return file_name, function_name, line_num


def get_line_num():
    return sys._getframe().f_lineno  # 当前行号


def save_dict_as_csv(filename, dict_need_save):
    with open(filename, "w", encoding="utf-8") as csv_file:
        # writer = csv.DictWriter(csv_file)
        writer = csv.writer(csv_file)
        for key, value in dict_need_save.items():
            writer.writerow([key, value])


def load_dict_from_csv(filename):
    read_dict = {}
    with open(filename, "r", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        read_dict = dict(reader)
        return read_dict


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def get_contour_from_mask(mask_array):
    if torch.is_tensor(mask_array):
        mask_array = mask_array.numpy()
    mask_coutour_array = np.array(mask_array)
    rgb_array = cv2.cvtColor(mask_coutour_array, cv2.COLOR_GRAY2RGB)
    contours, hierarchy = cv2.findContours(rgb_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb_array, contours, -1, (0, 0, 255), 1)
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2GRAY)  # 再次转成灰度图,因为要从灰度图转到二值图
    # 大于2的值都置为255,当预测结果出来的时候同样这样处理,然后进行loss计算
    ret, binary_contuor_map = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)
    return binary_contuor_map
