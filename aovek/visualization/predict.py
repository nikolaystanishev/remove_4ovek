import numpy as np
import json
import os
import imghdr
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from prettytable import PrettyTable

from aovek.network.network import YOLO
from aovek.utils.image_processing import ImageProcessing


class Predict(ImageProcessing):

    def __init__(self, config):
        super().__init__(config)

        self.network = YOLO(config)
        self.network.load_model()

        self.grid_size = config['label_info']['grid_size']

        self.train_folder = config['dataset']['dataset_images']['train_folder']
        self.validation_folder =\
            config['dataset']['dataset_images']['validation_folder']
        self.test_folder = config['dataset']['dataset_images']['test_folder']

        self.models = {'52': './model/52/model.h5'}

    def predict(self, image_file):
        image, _ = self.process_image(image_file)

        predict = self.network.predict_boxes(image)

        self.draw_rectangles(image[0], predict)

        return predict

    def predict_image(self, image_file):
        image, _ = self.process_image(image_file)

        predict = self.network.predict_boxes(image)

        self.draw_rectangles(image[0], predict)

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

    def draw_rectangles(self, image, lables):
        fig, ax = plt.subplots(1)

        ax.imshow(np.squeeze(image))

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

    predict = Predict(config)

    predict.make_predictions_for_optimizers()
