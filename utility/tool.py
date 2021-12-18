import sys
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from skimage.measure import label
from skimage.transform import resize
from skimage.morphology import dilation
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from typing import Optional, Union, List, Tuple
import itertools

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def move_to_device(var_tensor):
    return var_tensor.to(device, non_blocking=True)


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
    mask_coutour_array = np.array(mask_array, dtype='uint8')
    rgb_array = cv2.cvtColor(mask_coutour_array, cv2.COLOR_GRAY2RGB)
    contours, hierarchy = cv2.findContours(mask_coutour_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb_array, contours, -1, (0, 0, 255), 1)
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2GRAY)  # 再次转成灰度图,因为要从灰度图转到二值图
    # 大于2的值都置为255,当预测结果出来的时候同样这样处理,然后进行loss计算
    # 这样只会保留边界,其余的都是0
    ret, binary_contuor_map = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)
    return binary_contuor_map


def _edt_binary_mask(mask, resolution, alpha):
    if (mask == 1).all():  # tanh(5) = 0.99991
        return np.ones_like(mask).astype(float) * 5
    return distance_transform_edt(mask, resolution) / alpha


def get_distance_edt(binary_mask, alpha_fore: float = 8.0, alpha_back: float = 50.0):
    fore = (binary_mask != 0).astype(np.uint8)
    back = (binary_mask == 0).astype(np.uint8)
    resolution = (1.0, 1.0)
    fore_edt = _edt_binary_mask(fore, resolution, alpha_fore)
    back_edt = _edt_binary_mask(back, resolution, alpha_back)
    distance = fore_edt - back_edt
    return distance

# 调用此函数时直接 bcd_watershed(img)即可，img是模型输出的3通道图片
def bcd_watershed(img, thres1=0.9, thres2=0.8, thres3=0.85, thres4=0.5, thres5=0.0, thres_small=16,
                  scale_factors=(1.0, 1.0), remove_small_mode='background', seed_thres=32, return_seed=False):
    r"""Convert binary foreground probability maps, instance contours and signed distance
        transform to instance masks via watershed segmentation algorithm.

        Note:
            This function uses the `skimage.segmentation.watershed <https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/_watershed.py#L89>`_
            function that converts the input image into ``np.float64`` data type for processing. Therefore please make sure enough memory is allocated when handling large arrays.

        Args:
            img (numpy.ndarray): foreground and contour probability of shape :math:`(C, Y, X)`.
            thres1 (float): threshold of seeds. Default: 0.9
            thres2 (float): threshold of instance contours. Default: 0.8
            thres3 (float): threshold of foreground. Default: 0.85
            thres4 (float): threshold of signed distance for locating seeds. Default: 0.5
            thres5 (float): threshold of signed distance for foreground. Default: 0.0
            thres_small (int): size threshold of small objects to remove. Default: 16
            scale_factors (tuple): scale factors for resizing in :math:`( Y, X)` order. Default: ( 1.0, 1.0)
            remove_small_mode (str): ``'background'``, ``'neighbor'`` or ``'none'``. Default: ``'background'``
        """
    assert img.shape[0] == 3 # channel = 3
    semantic, boundary, distance = img[0], img[1], img[2]
    distance = (distance / 255.0) * 2.0 - 1.0

    seed_map = (semantic > int(255 * thres1)) * (boundary < int(255 * thres2)) * (distance > thres4)
    foreground = (semantic > int(255 * thres3)) * (distance > thres5)
    seed = label(seed_map)  # 生成连通图，用数字0-n表示，相通的标为同一类，不通的为不同类。这里默认参数似乎是判断上下左右4个像素是否联通。可选8像素
    seed = remove_small_objects(seed, seed_thres)  # 第二个是mini_size
    segm = watershed(-semantic.astype(np.float64), seed, mask=foreground)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x == 1.0 for x in scale_factors):
        target_size = (int(semantic.shape[0] * scale_factors[0]),
                       int(semantic.shape[1] * scale_factors[1]),
                       int(semantic.shape[2] * scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)

    if not return_seed:
        return cast2dtype(segm)

    return cast2dtype(segm), seed


def bc_watershed(volume, thres1=0.9, thres2=0.8, thres3=0.85, thres_small=128, scale_factors=(1.0, 1.0, 1.0),
                 remove_small_mode='background', seed_thres=32):
    r"""Convert binary foreground probability maps and instance contours to
    instance masks via watershed segmentation algorithm.

    Note:
        This function uses the `skimage.segmentation.watershed <https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/_watershed.py#L89>`_
        function that converts the input image into ``np.float64`` data type for processing. Therefore please make sure enough memory is allocated when handling large arrays.

    Args:
        volume (numpy.ndarray): foreground and contour probability of shape :math:`(C, Z, Y, X)`.
        thres1 (float): threshold of seeds. Default: 0.9
        thres2 (float): threshold of instance contours. Default: 0.8
        thres3 (float): threshold of foreground. Default: 0.85
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: (1.0, 1.0, 1.0)
        remove_small_mode (str): ``'background'``, ``'neighbor'`` or ``'none'``. Default: ``'background'``
    """
    assert volume.shape[0] == 2
    semantic = volume[0]
    boundary = volume[1]
    seed_map = (semantic > int(255 * thres1)) * (boundary < int(255 * thres2))
    foreground = (semantic > int(255 * thres3))
    seed = label(seed_map)
    seed = remove_small_objects(seed, seed_thres)
    segm = watershed(-semantic.astype(np.float64), seed, mask=foreground)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x == 1.0 for x in scale_factors):
        target_size = (int(semantic.shape[0] * scale_factors[0]),
                       int(semantic.shape[1] * scale_factors[1]),
                       int(semantic.shape[2] * scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)

    return cast2dtype(segm)


def remove_small_instances(segm: np.ndarray,
                           thres_small: int = 128,
                           mode: str = 'background'):
    """Remove small spurious instances.
    """
    assert mode in ['none',
                    'background',
                    'background_2d',
                    'neighbor',
                    'neighbor_2d']

    if mode == 'none':
        return segm

    if mode == 'background':
        return remove_small_objects(segm, thres_small)
    elif mode == 'background_2d':
        temp = [remove_small_objects(segm[i], thres_small)
                for i in range(segm.shape[0])]
        return np.stack(temp, axis=0)

    if mode == 'neighbor':
        return merge_small_objects(segm, thres_small, do_3d=True)
    elif mode == 'neighbor_2d':
        temp = [merge_small_objects(segm[i], thres_small)
                for i in range(segm.shape[0])]
        return np.stack(temp, axis=0)


def merge_small_objects(segm, thres_small, do_3d=False):
    struct = np.ones((1,3,3)) if do_3d else np.ones((3,3))
    indices, counts = np.unique(segm, return_counts=True)

    for i in range(len(indices)):
        idx = indices[i]
        if counts[i] < thres_small:
            temp = (segm == idx).astype(np.uint8)
            coord = bbox_ND(temp, relax=2)
            cropped = crop_ND(temp, coord)

            diff = dilation(cropped, struct) - cropped
            diff_segm = crop_ND(segm, coord)
            diff_segm[np.where(diff==0)]=0

            u, ct = np.unique(diff_segm, return_counts=True)
            if len(u) > 1 and u[0] == 0:
                u, ct = u[1:], ct[1:]

            segm[np.where(segm==idx)] = u[np.argmax(ct)]

    return segm


def cast2dtype(segm):
    """Cast the segmentation mask to the best dtype to save storage.
    """
    max_id = np.amax(np.unique(segm))
    m_type = getSegType(int(max_id))
    return segm.astype(m_type)


def getSegType(mid):
    # reduce the label dtype
    m_type = np.uint64
    if mid < 2**8:
        m_type = np.uint8
    elif mid < 2**16:
        m_type = np.uint16
    elif mid < 2**32:
        m_type = np.uint32
    return m_type


def crop_ND(img: np.ndarray, coord: Tuple[int]) -> np.ndarray:
    N = img.ndim
    assert len(coord) == N * 2
    slicing = []
    for i in range(N):
        slicing.append(slice(coord[2*i], coord[2*i+1]))
    slicing = tuple(slicing)
    return img[slicing].copy()


def bbox_ND(img: np.ndarray, relax: int = 0) -> tuple:
    """Calculate the bounding box of an object in a N-dimensional
    numpy array. All non-zero elements are treated as foregounrd.
    Reference: https://stackoverflow.com/a/31402351

    Args:
        img (np.ndarray): a N-dimensional array with zero as background.

    Returns:
        tuple: N-dimensional bounding box coordinates.
    """
    N = img.ndim
    out = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])

    return bbox_relax(out, img.shape, relax)


def bbox_relax(coord: Union[tuple, list],
               shape: tuple,
               relax: int = 0) -> tuple:

    assert len(coord) == len(shape) * 2
    coord = list(coord)
    for i in range(len(shape)):
        coord[2*i] = max(0, coord[2*i]-relax)
        coord[2*i+1] = min(shape[i], coord[2*i+1]+relax)

    return tuple(coord)