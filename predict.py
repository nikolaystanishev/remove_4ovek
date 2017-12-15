from scipy import ndimage
import numpy as np
from scipy.misc import imresize
import json
import os
import imghdr
from datetime import datetime

from network import YOLO


class Predict:

    def __init__(self, config):
        self.network = YOLO(config)
        self.network.load_model()

        self.image_size = config['image_info']['image_size']
        self.pixel_depth = config['image_info']['pixel_depth']
        self.color_channels = config['image_info']['color_channels']

        self.train_folder = config['dataset']['dataset_images']['train_folder']
        self.validation_folder =\
            config['dataset']['dataset_images']['validation_folder']
        self.test_folder = config['dataset']['dataset_images']['train_folder']

        self.optimizers = {
            '<class \'keras.optimizers.SGD\'>': './model/03/model.h5',
            '<class \'keras.optimizers.RMSprop\'>': './model/04/model.h5',
            '<class \'keras.optimizers.Adagrad\'>': './model/05/model.h5',
            '<class \'keras.optimizers.Adadelta\'>': './model/06/model.h5',
            '<class \'keras.optimizers.Adam\'>': './model/07/model.h5',
            '<class \'keras.optimizers.Adamax\'>': './model/08/model.h5',
            '<class \'keras.optimizers.Nadam\'>': './model/09/model.h5',
            '<class \'keras.optimizers.SGD\'> with changed params':
                './model/10/model.h5'}

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

    def make_predictions_for_optimizers(self):
        for optimizer, model_file in self.optimizers.items():
            start_time = datetime.now()

            self.network.load_model_file(model_file)
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(optimizer)
            print('==========================================================')
            print('Train')
            print('==========================================================')
            self.make_predictions_for_folder(self.train_folder)
            print('==========================================================')
            print('Validation')
            print('==========================================================')
            self.make_predictions_for_folder(self.validation_folder)
            print('==========================================================')
            print('Test')
            print('==========================================================')
            self.make_predictions_for_folder(self.test_folder)

            end_time = datetime.now()
            full_time = end_time - start_time

            print('==========================================================')
            print('Time: {}'.format(full_time))
            print('==========================================================')

    def make_predictions_for_folder(self, path):
        for dirpath, dirnames, filenames in os.walk(path):
            image_files = map(lambda filename: os.path.join(dirpath, filename),
                              filenames)

            for image_file in image_files:
                image_type = imghdr.what(image_file)
                if not image_type:
                    continue

                prediction = self.predict(image_file)

                for el in prediction[0]:
                    for e in el:
                        if np.sum(e) != 0:
                                print(e)


if __name__ == '__main__':
    with open('./config.json') as config_file:
        config = json.load(config_file)

    predict = Predict(config)

    predict.make_predictions_for_optimizers()
