from scipy import ndimage
import numpy as np
from scipy.misc import imresize
import json

from network import YOLO


class Predict:

    def __init__(self, config):
        self.model_pickle_file = config['network']['model_pickle_file']

        self.network = YOLO(config)
        self.network.load_model()

        self.image_size = config['image_info']['image_size']
        self.pixel_depth = config['image_info']['pixel_depth']
        self.color_channels = config['image_info']['color_channels']

    def predict(self, image_file):
        image = self.get_image_from_file(image_file)

        predict = self.network.predict(image)
        return predict

    def get_image_from_file(self, image_file):
        image_data = (ndimage.imread(image_file).astype(float) -
                      (self.pixel_depth / 2)) / self.pixel_depth

        if len(image_data.shape) != self.color_channels:
            image_data = np.repeat(image_data[:, :, np.newaxis],
                                   self.color_channels, axis=2)

        if image_data.shape != (self.image_size, self.image_size,
                                self.color_channels):
            image = imresize(image_data, size=(self.image_size,
                                               self.image_size))
        else:
            image = image_data

        image = np.expand_dims(image, axis=0)
        return image


if __name__ == '__main__':
    with open('./config.json') as config_file:
        config = json.load(config_file)

    predict = Predict(config)

    path = './dataset/cvpr10_multiview_pedestrians/test/00036.png'

    prediction = predict.predict(path)
    print(prediction)
