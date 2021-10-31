from utility.rle_tool import *
from utility.tool import *
import os


class TrainImageHandler(object):
    def __init__(self):
        self.img_dir = "/Users/wangyu/Desktop/利兹上课资料/Kaggle比赛/data/train";
        self.dict_imgID_image = {}
        self.save_img_dir = "/Users/wangyu/Desktop/利兹上课资料/Kaggle比赛/data/train_image.csv"
        data_exist_bool = os.path.exists(self.save_img_dir)

        if not data_exist_bool:
            self.get_dict_imgID_image()
            save_dict_as_csv(self.save_img_dir, self.dict_imgID_image)
        else:
            self.dict_imgID_image = load_dict_from_csv(self.save_img_dir)

    def __len__(self):
        return len(self.dict_imgID_mask)

    def get_dict_imgID_image(self):
        for root, dirs, files in os.walk(self.img_dir, topdown=True):
            for image_name in files:
                if '.png' in image_name:
                    image_full_path = os.path.join(self.img_dir, image_name)
                    img = cv2.imread(image_full_path, 0)
                    image_id = image_name.split('.')[0]
                    self.dict_imgID_image[image_id] = img



