import sys
import csv
import cv2
import matplotlib.pyplot as plt


def outclude_hidden_files(files):
    return [f for f in files if not f[0] == '.']


def outclude_hidden_dirs(dirs):
    return [d for d in dirs if not d[0] == '.']


def show_image(image, window_name='test'):
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# encoding=utf-8
def get_cur_info():
    file_name = sys._getframe().f_code.co_filename  # 当前文件名，可以通过__file__获得
    function_name = sys._getframe().f_code.co_name  # 当前函数名
    line_num = get_line_num()
    return file_name, function_name, line_num


def get_line_num():
    return sys._getframe().f_lineno  # 当前行号


def save_dict_as_csv(filename, dict_need_save):
    with open(filename, "w") as csv_file:
        # writer = csv.DictWriter(csv_file)
        writer = csv.writer(csv_file)
        for key, value in dict_need_save.items():
            writer.writerow([key, value])


def load_dict_from_csv(filename):
    read_dict = {}
    with open(filename, "r") as csv_file:
        reader = csv.reader(csv_file)
        read_dict = dict(reader)
        return read_dict
