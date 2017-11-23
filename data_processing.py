import os
import numpy as np
from scipy import ndimage
from scipy.misc import imresize
import xml.etree.ElementTree as ET
import pickle
from PIL import Image
from datetime import datetime
import json


class DataProcessing:
    """
        Class for processing data and loading it to pickle file
    """

    def __init__(self, config):
        self.train_folder = config['dataset']['dataset_images']['train_folder']
        self.validation_folder =\
            config['dataset']['dataset_images']['validation_folder']
        self.test_folder = config['dataset']['dataset_images']['train_folder']

        self.train_annotations =\
            config['dataset']['dataset_annotations']['train_annotations']
        self.validation_annotations =\
            config['dataset']['dataset_annotations']['validation_annotations']
        self.test_annotations =\
            config['dataset']['dataset_annotations']['train_annotations']

        self.image_size = config['image_info']['image_size']
        self.pixel_depth = config['image_info']['pixel_depth']
        self.color_channels = config['image_info']['color_channels']

        self.time = None

    def pickle_data(self, pickled_name):
        start_time = datetime.now()

        dataset = self.get_dataset()

        with open(pickled_name, 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

        end_time = datetime.now()
        self.time = end_time - start_time

    def get_dataset(self):
        dataset = {}

        train_data, train_labels =\
            self.get_images_data_from_path(self.train_folder,
                                           self.train_annotations)
        validation_data, validation_labels =\
            self.get_images_data_from_path(self.validation_folder,
                                           self.validation_annotations)
        test_data, test_labels =\
            self.get_images_data_from_path(self.test_folder,
                                           self.test_annotations)

        dataset.update({'train_data': train_data,
                        'train_labels': train_labels})
        dataset.update({'validation_data': validation_data,
                        'validation_labels': validation_labels})
        dataset.update({'test_data': test_data, 'test_labels': test_labels})

        return dataset

    def get_images_data_from_path(self, path, annotations):
        images = np.ndarray(shape=(0, self.image_size, self.image_size,
                                   self.color_channels), dtype=np.float32)
        labels = np.ndarray(0, dtype=np.int32)

        for dirpath, dirnames, filenames in os.walk(path):
            if not dirnames:
                images_info = self.get_info_for_images(annotations)
                image_files = self.get_images_files_path(dirpath, filenames)
                images_segment, labels_segment =\
                    self.get_images_and_labels(image_files, images_info)
                images = np.concatenate((images, images_segment))
                labels = np.append(labels, labels_segment)
        return images, labels

    def get_info_for_images(self, annotations):
        images_info = {}

        for annotation in annotations:
            tree = ET.parse(annotation)
            root = tree.getroot()
            for child in root:
                image_name =\
                    child.find('image').find('name').text.split('/')[1]
                image_center =\
                    (int(child.find('annorect').find('objpos').find('x').text),
                     int(child.find('annorect').find('objpos').find('y').text))
                image_rect = ((int(child.find('annorect').find('x1').text),
                               int(child.find('annorect').find('y1').text)),
                              (int(child.find('annorect').find('x2').text),
                               int(child.find('annorect').find('y2').text)))
                images_info[image_name] = (image_center, image_rect)

        return images_info

    def get_images_files_path(self, dirpath, filenames):
        image_files = map(lambda filename: os.path.join(dirpath, filename),
                          filenames)
        return image_files

    def get_images_and_labels(self, image_files, images_info):
        images = np.ndarray(shape=(0, self.image_size, self.image_size,
                                   self.color_channels), dtype=np.float32)
        labels = np.ndarray(shape=(0, 4), dtype=np.int32)

        for image_file in image_files:
            image, original_size = self.process_image(image_file)

            image_info = images_info[os.path.basename(image_file)]
            label = self.process_image_labels(image_info, original_size)

            images = np.concatenate((images, image))
            labels = np.append(labels, label)
        return images, labels

    def process_image(self, image_file):
        image_data = (ndimage.imread(image_file).astype(float) -
                      (self.pixel_depth / 2)) / self.pixel_depth

        original_size = image_data.shape

        if len(image_data.shape) == 2:
            image_data = np.stack((image_data,) * 3)

        if image_data.shape != (self.image_size, self.image_size,
                                self.color_channels):
            image = imresize(image_data, size=(self.image_size,
                                               self.image_size))
        else:
            image = image_data
        image = np.expand_dims(image, axis=0)
        return (image, original_size)

    def process_image_labels(self, annotation, original_size):
        center_x = annotation[0][0] / original_size[0]
        center_y = annotation[0][1] / original_size[1]

        center_w =\
            (annotation[1][1][0] - annotation[1][0][0]) / original_size[0]
        center_h =\
            (annotation[1][1][1] - annotation[1][0][1]) / original_size[1]

        return (center_x, center_y, center_w, center_h)

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
    dp.pickle_data('./dataset/dataset.pickle')

    print('Taken time: {}'.format(str(dp.get_time())))
