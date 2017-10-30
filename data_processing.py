import os
import numpy as np
from scipy import ndimage
from scipy.misc import imresize
import csv
import pickle

from settings import image_size, pixel_depth, color_channels,\
    number_of_train_images, number_of_test_images


class DataProcessing:
    """
    Class for processing and loading it to pickle file
    """

    def __init__(self, train_folder, test_folder):
        self.train_folder = train_folder
        self.test_folder = test_folder

    def get_training(self):
        images = np.ndarray(shape=(number_of_train_images, image_size,
                                   image_size, color_channels),
                            dtype=np.float32)
        labels = np.ndarray(number_of_train_images, dtype=np.int32)
        num_images = 0
        image_categories_dir = os.listdir(self.train_folder)
        for directory in image_categories_dir:
            directory_path = os.path.join(self.train_folder, directory)
            image_files = os.listdir(directory_path)
            for image_filename in image_files:
                image_path = os.path.join(directory_path, image_filename)
                try:
                    image_data = (ndimage.imread(image_path).astype(float) -
                                  (pixel_depth / 2)) / pixel_depth
                    if image_data.shape != (image_size, image_size):
                        image = imresize(image_data,
                                         size=(image_size, image_size))
                    else:
                        image = image_data
                    images[num_images, :, :, :] = image
                    labels[num_images] = int(directory)
                    num_images = num_images + 1
                except OSError as e:
                    pass
        images = images[0:num_images, :, :, :]
        labels = labels[0:num_images]
        return {'train_data': images, 'train_labels': labels}

    def get_test(self):
        images = np.ndarray(shape=(number_of_test_images, image_size,
                                   image_size, color_channels),
                            dtype=np.float32)
        labels = np.ndarray(number_of_train_images, dtype=np.int32)
        num_images = 0
        image_files = os.listdir(self.test_folder)
        images_info = {}
        with open(os.path.join(self.test_folder, 'images_info.csv'),
                  'rt') as cf:
            reader = csv.reader(cf)
            next(reader, None)
            images_info = {row[0].split(';')[0]: row[0].split(';')[-1]
                           for row in reader}
        for image_filename in image_files:
            image_path = os.path.join(self.test_folder, image_filename)
            try:
                image_data = (ndimage.imread(image_path).astype(float) -
                              (pixel_depth / 2)) / pixel_depth
                if image_data.shape != (image_size, image_size):
                    image = imresize(image_data, size=(image_size, image_size))
                else:
                    image = image_data
                images[num_images, :, :, :] = image
                labels[num_images] = int(images_info[image_filename])
                num_images = num_images + 1
            except OSError as e:
                pass
        images = images[0:num_images, :, :, :]
        labels = labels[0:num_images]
        return {'test_data': images, 'test_labels': labels}

    def pickle_data(self, pickled_name):
        dataset = {}
        train = self.get_training()
        test = self.get_test()
        dataset.update(train)
        dataset.update(test)
        with open(pickled_name, 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    dp = DataProcessing('./dataset/train_images', './dataset/test_images')
    dp.pickle_data('./dataset/dataset.pickle')
