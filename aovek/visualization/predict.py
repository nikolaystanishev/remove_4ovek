from scipy import ndimage
import numpy as np
from skimage.transform import resize
import json
import os
import imghdr
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from prettytable import PrettyTable
import tensorflow as tf

from aovek.network.network import YOLO


class Predict:

    def __init__(self, config):
        self.network = YOLO(config)

        self.image_size = config['image_info']['image_size']
        self.pixel_depth = config['image_info']['pixel_depth']
        self.color_channels = config['image_info']['color_channels']

        self.grid_size = config['label_info']['grid_size']

        self.train_folder = config['dataset']['dataset_images']['train_folder']
        self.validation_folder =\
            config['dataset']['dataset_images']['validation_folder']
        self.test_folder = config['dataset']['dataset_images']['test_folder']

        self.models = {'49': './model/49/model.h5'}

    def predict(self, image_file):
        image = self.get_image_from_file(image_file)

        predict = self.network.predict(image)

        self.draw_rectangles(image[0], predict)

        return predict

    def get_image_from_file(self, image_file):
        image_data = ndimage.imread(image_file).astype(float)

        if len(image_data.shape) != self.color_channels:
            image_data = np.repeat(image_data[:, :, np.newaxis],
                                   self.color_channels, axis=2)

        if image_data.shape != (self.image_size, self.image_size,
                                self.color_channels):
            image = resize(image_data,
                           output_shape=(self.image_size,
                                         self.image_size),
                           mode='constant')
        else:
            image = image_data

        image = np.expand_dims(image, axis=0)

        image = self.normalize_image_from_0_to_1(image)

        return image

    def make_predictions_for_optimizers(self):
        for optimizer, model_file in self.models.items():
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
            print('==========================================================')
            print('VOC')
            print('==========================================================')
            self.make_predictions_for_folder('../../../../Downloads/dataset/'
                                             'VOCdevkit/VOC2012/JPEGImages/')
            print('==========================================================')

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

                if np.sum(prediction) != 0:
                    print('.')

                for el in prediction:
                    if np.sum(el) != 0:
                            print(el)

    def normalize_image_from_minus1_to_1(self, image):
        normalized_image = (image - (self.pixel_depth / 2)) / self.pixel_depth

        return normalized_image

    def normalize_image_from_0_to_1(self, image):
        normalized_image = image / self.pixel_depth

        return normalized_image

    def normalize_image_without_normalization(self, image):
        normalized_image = image

        return normalized_image

    def draw_rectangles(self, image, lables):
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for label in lables:
            print(label[4])

            x = label[0] * self.image_size
            y = label[1] * self.image_size
            w = (label[2] - label[0]) * self.image_size
            h = (label[3] - label[1]) * self.image_size

            rect = Rectangle((x, y), w, h, linewidth=1,
                             edgecolor='r', facecolor='none')

            ax.add_patch(rect)

        plt.show()

    def draw_grid(self, image, predict):
        predict_table = PrettyTable()

        for p in predict:
            predict_table.add_row(p)
        print(predict)
        print(predict_table)

        fig, ax = plt.subplots(1)
        dx = int(self.image_size / self.grid_size)

        grid_color = [0, 0, 0]

        image[0][:, ::dx, :] = grid_color
        image[0][::dx, :, :] = grid_color
        ax.imshow(image[0])

        plt.show()


if __name__ == '__main__':
    with open('./config.json') as config_file:
        config = json.load(config_file)

    with tf.Session():
        predict = Predict(config)

        predict.make_predictions_for_optimizers()
