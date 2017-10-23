import os
import numpy as np
from scipy import ndimage
from scipy.misc import imresize
import csv
import pickle

from settings import image_size, pixel_depth, color_channels,\
    number_of_train_images, number_of_test_images


def get_training(folder):
    images = np.ndarray(shape=(number_of_train_images, image_size,
                               image_size, color_channels),
                        dtype=np.float32)
    labels = []
    num_images = 0
    image_categories_dir = os.listdir(folder)
    for directory in image_categories_dir:
        directory_path = os.path.join(folder, directory)
        image_files = os.listdir(directory_path)
        for image_filename in image_files:
            image_path = os.path.join(directory_path, image_filename)
            try:
                image_data = (ndimage.imread(image_path).astype(float) -
                              (pixel_depth / 2)) / pixel_depth
                if image_data.shape != (image_size, image_size):
                    image = imresize(image_data, size=(image_size, image_size))
                else:
                    image = image_data
                images[num_images, :, :, :] = image
                num_images = num_images + 1
                label = int(directory)
                labels.append(label)
            except OSError as e:
                pass
    images = images[0:num_images, :, :, :]
    return {'train_data': images, 'train_labels': labels}


def get_test(folder):
    images = np.ndarray(shape=(number_of_test_images, image_size,
                               image_size, color_channels),
                        dtype=np.float32)
    labels = []
    num_images = 0
    image_files = os.listdir(folder)
    images_info = {}
    with open(os.path.join(folder, 'images_info.csv'), 'rt') as cf:
        reader = csv.reader(cf)
        next(reader, None)
        images_info = {row[0].split(';')[0]: row[0].split(';')[-1]
                       for row in reader}
    for image_filename in image_files:
        image_path = os.path.join(folder, image_filename)
        try:
            image_data = (ndimage.imread(image_path).astype(float) -
                          (pixel_depth / 2)) / pixel_depth
            if image_data.shape != (image_size, image_size):
                image = imresize(image_data, size=(image_size, image_size))
            else:
                image = image_data
            images[num_images, :, :, :] = image
            num_images = num_images + 1
            label = int(images_info[image_filename])
            labels.append(label)
        except OSError as e:
            pass
    images = images[0:num_images, :, :, :]
    return {'test_data': images, 'test_labels': labels}


def pickle_data(train_folder, test_folder, pickled_name):
    dataset = {}
    train = get_training(train_folder)
    test = get_test(test_folder)
    dataset.update(train)
    dataset.update(test)
    with open(pickled_name, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    pickle_data('./dataset/train_images', './dataset/test_images',
                './dataset/dataset.pickle')
