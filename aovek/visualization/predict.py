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

    def predict(self, image_file):
        image, original_size = self.process_image(image_file)
        original_image = self.load_image(image_file)

        predict = self.network.predict_boxes(image)
        predict = self.network.sess_run(predict)

        self.draw_rectangles(original_image, original_size, predict)

        return predict

    def predict_all_boxes(self, image_file):
        image, original_size = self.process_image(image_file)
        original_image = self.load_image(image_file)

        predict = self.network.predict(image)

        self.draw_rectangles(original_image, original_size,
                             predict, all_boxes=True)

        return predict

    def predict_video(self, video):
        predictions = self.network.predict_video(video)

        return predictions

    def make_predictions_for_datasets(self):
        start_time = datetime.now()

        self.make_predictions_for_dataset('Train', self.train_folder)
        self.make_predictions_for_dataset('Validation', self.validation_folder)
        self.make_predictions_for_dataset('Test', self.test_folder)

        end_time = datetime.now()
        full_time = end_time - start_time

        print('==========================================================')
        print('Time: {}'.format(full_time))
        print('==========================================================')

    def make_predictions_for_dataset(self, dataset, dataset_path):
        print('==========================================================')
        print(dataset)
        print('==========================================================')
        self.make_predictions_for_folder(dataset_path)

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

    def draw_rectangles(self, image, original_size, lables, all_boxes=False):
        fig, ax = plt.subplots(1)

        ax.imshow(np.squeeze(image))

        for label in lables:
            if all_boxes is False:
                print(label[4])

            x = label[0] * original_size[0]
            y = label[1] * original_size[1]
            w = (label[2] - label[0]) * original_size[0]
            h = (label[3] - label[1]) * original_size[1]

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

    predict.make_predictions_for_datasets()
