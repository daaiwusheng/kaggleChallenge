from dataprocess.train_labels_process import *
from dataprocess.train_image_process import *
from dataprocess.kaggle_data_provider import *
from dataprocess.write_patches import *
from utility.rle_tool import *
import cv2

dataprovider = KaggleDataSaver(256)

imagename = dataprovider.train_labels[0]
img = cv2.imread(imagename)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

image_test_array = np.array(img)

# cv2.drawContours(image_test_array, contours, -1, (0, 0, 255), 1)
rgb_array = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
cv2.drawContours(rgb_array, contours, -1, (0, 0, 255), 1)
# cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
print(rgb_array.shape)

gray = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)
cv2.imshow("img", binary)
cv2.waitKey(0)

print(image_test_array.shape)
gray = cv2.cvtColor(image_test_array, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)
print(binary)

cv2.imshow("img", binary)
cv2.waitKey(0)
# print(image_test_array)

#
# cv2.imwrite("/Users/wangyu/Downloads/test.png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
#
# cv2.imshow("img", img)
# cv2.waitKey(0)

# imagename = "/Users/wangyu/Downloads/test.png"
# img = cv2.imread(imagename)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)
# print(binary)
