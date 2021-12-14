from dataprocess.train_labels_process import *
from dataprocess.train_image_process import *
from dataprocess.kaggle_data_provider import *
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_holes
from dataprocess.write_patches import *
from utility.rle_tool import *
import cv2
import matplotlib.pyplot as plt

dataprovider = KaggleDataSaver(256)

imagename = dataprovider.train_labels[0]
img = cv2.imread(imagename)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)

edt = distance_transform_edt(binary)
# print(np.max(edt))
# cv2.imshow("img", edt)
# cv2.waitKey(0)
plt.imshow(edt, cmap='gray_r')
plt.show()
