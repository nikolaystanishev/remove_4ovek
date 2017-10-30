import os
import numpy as np
from scipy import ndimage
from scipy.misc import imresize
import csv
import pickle
from PIL import Image
from datetime import datetime

from settings import pixel_depth, color_channels, name_of_info_file


class DataProcessing:
    """
    Class for processing data and loading it to pickle file
    """

    def __init__(self, train_folder, test_folder):
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.image_size = self.get_average_image_size(self.train_folder)
        self.time = None

    def pickle_data(self, pickled_name):
        start_time = datetime.now()
        dataset = {}
        train_data, train_labels =\
            self.get_images_data_from_path(self.train_folder)
        test_data, test_labels =\
            self.get_images_data_from_path(self.test_folder)
        dataset.update({'train_data': train_data,
                        'train_labels': train_labels})
        dataset.update({'test_data': test_data, 'test_labels': test_labels})
        with open(pickled_name, 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        end_time = datetime.now()
        self.time = end_time - start_time

    def get_images_data_from_path(self, path):
        images = np.ndarray(shape=(0, self.image_size, self.image_size,
                                   color_channels), dtype=np.float32)
        labels = np.ndarray(0, dtype=np.int32)

        for dirpath, dirnames, filenames in os.walk(path):
            if not dirnames:
                images_info = self.get_info_for_images(dirpath)
                image_files = self.get_images_files_path(dirpath, filenames)
                images_segment, labels_segment =\
                    self.get_images_and_labels(image_files, images_info)
                images = np.concatenate((images, images_segment))
                labels = np.append(labels, labels_segment)
        return images, labels

    def get_info_for_images(self, dirpath):
        images_info = {}
        with open(os.path.join(dirpath, name_of_info_file),
                  'rt') as cf:
            reader = csv.reader(cf)
            next(reader, None)
            images_info = {row[0].split(';')[0]: row[0].split(';')[-1]
                           for row in reader}
        return images_info

    def get_images_files_path(self, dirpath, filenames):
        filenames.remove(name_of_info_file)
        image_files = map(lambda filename: os.path.join(dirpath, filename),
                          filenames)
        return image_files

    def get_images_and_labels(self, image_files, images_info):
        images = np.ndarray(shape=(0, self.image_size, self.image_size,
                                   color_channels), dtype=np.float32)
        labels = np.ndarray(0, dtype=np.int32)

        for image_file in image_files:
            image = self.process_image(image_file)
            label = int(images_info[os.path.basename(image_file)])
            images = np.concatenate((images, image))
            labels = np.append(labels, label)
        return images, labels

    def process_image(self, image_file):
        image_data = (ndimage.imread(image_file).astype(float) -
                      (pixel_depth / 2)) / pixel_depth
        if image_data.shape != (self.image_size, self.image_size):
            image = imresize(image_data, size=(self.image_size,
                                               self.image_size))
        else:
            image = image_data
        image = np.expand_dims(image, axis=0)
        return image

    def get_average_image_size(self, path):
        resolutions = []
        for dirpath, dirnames, filenames in os.walk(path):
            if not dirnames:
                filenames.remove(name_of_info_file)
                for filename in filenames:
                    img = Image.open(os.path.join(dirpath, filename))
                    resolutions.append(img.size)
        mean = int(np.mean(resolutions))
        return mean

    def get_time(self):
        return self.time


if __name__ == '__main__':
    dp = DataProcessing('./dataset/train_images', './dataset/test_images')
    dp.pickle_data('./dataset/dataset.pickle')
    print('Taken time: {}'.format(str(dp.get_time())))
