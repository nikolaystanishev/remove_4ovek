import os
import numpy as np
from scipy import ndimage
from skimage.transform import resize
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
            config['dataset']['dataset_annotations']['test_annotations']

        self.dataset_folder = config['dataset']['folder']

        self.image_size = config['image_info']['image_size']
        self.pixel_depth = config['image_info']['pixel_depth']
        self.color_channels = config['image_info']['color_channels']

        self.train_pickle_name = config['dataset']['pickle_name']['train']
        self.validation_pickle_name =\
            config['dataset']['pickle_name']['validation']
        self.test_pickle_name = config['dataset']['pickle_name']['test']

        self.grid_size = config['label_info']['grid_size']
        self.number_of_annotations =\
            config['label_info']['number_of_annotations']

        self.normalizer = None

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

    def generate_dataset(self):
        for train_data in self.get_train():
            self.update_pickle(self.train_pickle_name, train_data)

        for validation_data in self.get_validation():
            self.update_pickle(self.validation_pickle_name, validation_data)

        for test_data in self.get_test():
            self.update_pickle(self.test_pickle_name, test_data)

    def update_pickle(self, pickle_file, data):
        print('\n+')
        with open(pickle_file, 'rb') as f:
            dataset = pickle.load(f)
            for k, v in data.items():
                dataset[k] = np.concatenate((dataset[k], v))

        with open(pickle_file, 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        print('\\/')

    def get_train(self):
        for annotation in self.train_annotations:
            train_data, train_labels = self.process_annotation(annotation)
            yield ({'data': train_data, 'labels': train_labels})

    def get_validation(self):
        for annotation in self.validation_annotations:
            validation_data, validation_labels =\
                self.process_annotation(annotation)
            yield ({'data': validation_data,
                    'labels': validation_labels})

    def get_test(self):
        for annotation in self.test_annotations:
            test_data, test_labels = self.process_annotation(annotation)
            yield ({'data': test_data, 'labels': test_labels})

    def process_annotation(self, annotation):
        print("*")
        images, labels = self.get_segment(annotation)

        return images, labels

    def get_segment(self, annotation):
        images_info = self.get_info_for_images(annotation)
        image_files = self.get_images_path_from_images_info(images_info)
        images, labels =\
            self.get_images_and_labels(image_files, images_info)

        return images, labels

    def get_info_for_images(self, annotation):
        images_info = {}

        tree = ET.parse(annotation)
        root = tree.getroot()
        for child in root:
            image_name = child.find('image').find('name').text

            images_info[image_name] = self.get_image_info_for_one_image(child)

        return images_info

    def get_image_info_for_one_image(self, xml_tag):
        image_info = []

        for image_rect_points in xml_tag.iter('annorect'):
            image_rect = ((int(image_rect_points.find('x1').text),
                           int(image_rect_points.find('y1').text)),
                          (int(image_rect_points.find('x2').text),
                           int(image_rect_points.find('y2').text)))

            try:
                image_center =\
                    (int(image_rect_points.find('objpos').find('x').text),
                     int(image_rect_points.find('objpos').find('y').text))
            except AttributeError as e:
                image_center = (((image_rect[1][0] - image_rect[0][0]) / 2 +
                                 image_rect[0][0]),
                                (((image_rect[1][1] - image_rect[0][1])) / 2 +
                                 image_rect[0][1]))
            image_info.append([image_center, image_rect])

        return image_info

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

    def process_image(self, image_file):
        image_data = ndimage.imread(image_file).astype(float)

        if len(image_data.shape) == 2:
            image_data = np.repeat(image_data[:, :, np.newaxis], 3, axis=2)

        original_size = image_data.shape[:-1][::-1]

        if image_data.shape != (self.image_size, self.image_size,
                                self.color_channels):
            image = resize(image_data,
                           output_shape=(self.image_size, self.image_size),
                           mode='constant')
        else:
            image = image_data

        image = np.expand_dims(image, axis=0)

        if self.normalizer == 0 or self.normalizer is None:
            image = self.normalize_image_without_normalization(image)
        elif self.normalizer == 1:
            image = self.normalize_image_from_0_to_1(image)
        elif self.normalizer == 2:
            image = self.normalize_image_from_minus1_to_1(image)

        return (image, original_size)

    def normalize_image_from_minus1_to_1(self, image):
        normalized_image = (image - (self.pixel_depth / 2)) / self.pixel_depth

        return normalized_image

    def normalize_image_from_0_to_1(self, image):
        normalized_image = image / self.pixel_depth

        return normalized_image

    def normalize_image_without_normalization(self, image):
        normalized_image = image

        return normalized_image

    def process_image_labels(self, annotations, original_size):
        label = np.zeros((self.grid_size, self.grid_size,
                          (1 + self.number_of_annotations)))

        for annotation in annotations:
            box, grid_x, grid_y =\
                self.process_label_annotation(annotation, original_size)

            if grid_x >= self.grid_size:
                grid_x = self.grid_size - 1
            if grid_y >= self.grid_size:
                grid_y = self.grid_size - 1

            label[grid_x, grid_y, 0:4] = box
            label[grid_x, grid_y, 4] = 1

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

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer


if __name__ == '__main__':
    with open('./config.json') as config_file:
        config = json.load(config_file)

    dp = DataProcessing(config)
    dp.set_normalizer(1)
    dp.pickle_dataset()

    print('Taken time: {}'.format(str(dp.get_time())))
