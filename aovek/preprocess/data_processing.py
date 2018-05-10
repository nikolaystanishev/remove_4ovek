import os
import numpy as np
import pickle
from PIL import Image
from datetime import datetime
import json

from aovek.utils.image_processing import ImageProcessing


class DataProcessing(ImageProcessing):
    """
        Class for processing data and loading it to pickle file
    """

    def __init__(self, config):
        super().__init__(config)

        dataset = config['dataset']['dataset']

        self.train_folder =\
            config['dataset'][dataset]['dataset_images']['train_folder']
        self.validation_folder =\
            config['dataset'][dataset]['dataset_images']['validation_folder']
        self.test_folder =\
            config['dataset'][dataset]['dataset_images']['train_folder']

        self.train_annotations =\
            config['dataset'][dataset]['dataset_annotations']['train_'
                                                              'annotations']
        self.validation_annotations =\
            config['dataset'][dataset]['dataset_annotations']['validation_'
                                                              'annotations']
        self.test_annotations =\
            config['dataset'][dataset]['dataset_annotations']['test_'
                                                              'annotations']

        self.dataset_folder = config['dataset'][dataset]['folder']

        self.train_pickle_name =\
            config['dataset'][dataset]['pickle_name']['train']
        self.validation_pickle_name =\
            config['dataset'][dataset]['pickle_name']['validation']
        self.test_pickle_name =\
            config['dataset'][dataset]['pickle_name']['test']

        self.grid_size = config['label_info']['grid_size']
        self.number_of_annotations =\
            config['label_info']['number_of_annotations']

        self.time = None

    def pickle_dataset(self):
        start_time = datetime.now()

        self.create_pickle()

        self.generate_dataset()

        end_time = datetime.now()
        self.time = end_time - start_time

    def create_pickle(self):
        self.write_start_values(self.train_pickle_name)
        self.write_start_values(self.validation_pickle_name)
        self.write_start_values(self.test_pickle_name)

    def write_start_values(self, pickle_file):
        with open(pickle_file, 'wb') as f:
            dataset_template =\
                {'data':
                    np.ndarray(shape=(0, self.image_size, self.image_size,
                                      self.color_channels), dtype=np.float32),
                 'labels': np.ndarray(shape=(0, self.grid_size, self.grid_size,
                                             (1 + self.number_of_annotations)),
                                      dtype=np.float32)}
            pickle.dump(dataset_template, f, pickle.HIGHEST_PROTOCOL)

    def update_pickle(self, pickle_file, data):
        print('\n+')
        with open(pickle_file, 'rb') as f:
            dataset = pickle.load(f)
            for k, v in data.items():
                dataset[k] = np.concatenate((dataset[k], v))

        with open(pickle_file, 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        print('\\/')

    def get_images_path_from_images_info(self, images_info):
        image_files = [image_path for image_path, _ in images_info.items()]

        return image_files

    def get_images_and_labels(self, image_files, images_info):
        images = np.ndarray(shape=(0, self.image_size, self.image_size,
                                   self.color_channels), dtype=np.float32)
        labels = np.ndarray(shape=(0, self.grid_size, self.grid_size,
                                   (1 + self.number_of_annotations)),
                            dtype=np.float32)

        for image_file in image_files:
            print(".", end='')
            image, original_size =\
                self.process_image(os.path.join(self.dataset_folder,
                                                image_file))

            image_info = images_info[image_file]
            label = self.process_image_labels(image_info, original_size)

            images = np.concatenate((images, image))
            labels = np.concatenate((labels, label))
        return images, labels

    def process_image_labels(self, annotations, original_size):
        label = np.zeros((self.grid_size, self.grid_size,
                          (1 + self.number_of_annotations)))

        for annotation in annotations:
            box, grid_x, grid_y =\
                self.process_label_annotation(annotation, original_size)

            if grid_x > self.grid_size:
                grid_x = self.grid_size - 1
            if grid_y > self.grid_size:
                grid_y = self.grid_size - 1

            label[grid_x - 1, grid_y - 1, 0:4] = box
            label[grid_x - 1, grid_y - 1, 4] = 1

        return np.array(label, ndmin=4)

    def process_label_annotation(self, annotation, original_size):
        center_x = annotation[0][0] / original_size[0]
        center_y = annotation[0][1] / original_size[1]

        center_w =\
            (annotation[1][1][0] - annotation[1][0][0]) / original_size[0]
        center_h =\
            (annotation[1][1][1] - annotation[1][0][1]) / original_size[1]

        box = np.array([center_x, center_y, center_w, center_h], ndmin=2)

        grid_x = int(center_x * self.grid_size)
        grid_y = int(center_y * self.grid_size)

        return box, grid_x, grid_y

    def get_average_image_size(self, path):
        resolutions = []
        for dirpath, dirnames, filenames in os.walk(path):
            if not dirnames:
                for filename in filenames:
                    img = Image.open(os.path.join(dirpath, filename))
                    resolutions.append(img.size)
        mean = int(np.mean(resolutions))
        print('Mean Image Size: {}'.format(mean))
        return mean

    def get_time(self):
        return self.time


if __name__ == '__main__':
    with open('./config.json') as config_file:
        config = json.load(config_file)

    dp = DataProcessing(config)
    dp.pickle_dataset()

    print('Taken time: {}'.format(str(dp.get_time())))
