from dataprocess.train_labels_process import *
from dataprocess.train_image_process import *
from dataprocess.kaggle_data_provider import *
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_holes
from dataprocess.write_patches import *
from utility.rle_tool import *

import matplotlib
import cv2
import matplotlib.pyplot as plt
from config import *
# dataprovider = KaggleDataSaver(256)
#
# imagename = dataprovider.train_labels[0]
# img = cv2.imread(imagename)
#
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)
#
# edt = distance_transform_edt(binary)
# # print(np.max(edt))
# # cv2.imshow("img", edt)
# # cv2.waitKey(0)
# plt.imshow(edt, cmap='gray_r')
# plt.show()

# label_provider = TrainLabelsProcessor()
# dict_imageID_label = label_provider.dict_imgID_mask
# i = 0  # for test, then only use one image
# for img_id, label in dict_imageID_label.items():
#     i += 1
#     label_array = np.array(label)
#
#     matplotlib.image.imsave(img_id + '.png', label_array)

input_file_path = "F:/LeedsDocs/Kaggle/train"

dataset = rasterio.open(input_file_path, 'r')
bands = [1, 2, 3]
data = dataset.read(bands)
transform = rasterio.transform.from_bounds(west, south, east, north, data.shape[1], data.shape[2])
crs = {'init': 'epsg:3006'}

with rasterio.open(output_file_path, 'w', driver='GTiff',
                   width=data.shape[1], height=data.shape[2],
                   count=3, dtype=data.dtype, nodata=0,
                   transform=transform, crs=crs) as dst:
    dst.write(data, indexes=bands)