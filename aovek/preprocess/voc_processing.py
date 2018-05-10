import os
import xml.etree.ElementTree as ET
import json
import random
import pickle

from aovek.preprocess.data_processing import DataProcessing


class VOCProcessing(DataProcessing):

    def __init__(self, config):
        super().__init__(config)

    def generate_dataset(self):
        images_info = self.get_info_for_images(self.train_annotations[0])

        images_info_random_keys = list(images_info.keys())
        random.shuffle(images_info_random_keys)
        images_info = {k: images_info[k] for k in images_info_random_keys}

        test_images_info, images_info =\
            self.get_images_info_segment(images_info, 500)
        self.generate_dataset_part(test_images_info, self.test_pickle_name)

        validation_images_info, images_info =\
            self.get_images_info_segment(images_info, 500)
        self.generate_dataset_part(validation_images_info,
                                   self.validation_pickle_name)

        train_images_info = images_info
        self.generate_dataset_part(train_images_info,
                                   self.train_pickle_name)

    def get_images_info_segment(self, images_info, size):
        images_name = list(images_info.keys())
        images_info_segment = {k: images_info[k] for k in images_name[:size]}
        images_info = {k: images_info[k] for k in images_name[size:]}

        return images_info_segment, images_info

    def generate_dataset_part(self, images_info, pickle_name):
        with open(pickle_name.split('/')[-1].split('.')[0] + 'images.pickle',
                  'wb') as f:
            dataset_template =\
                {'images': list(images_info.keys())}
            pickle.dump(dataset_template, f, pickle.HIGHEST_PROTOCOL)

        for test_data in self.get_segment(images_info):
            self.update_pickle(pickle_name, test_data)

    def get_segment(self, images_info):
        for images_info_segment in self.image_info_generator(images_info):
            image_files =\
                self.get_images_path_from_images_info(images_info_segment)
            images, labels =\
                self.get_images_and_labels(image_files, images_info_segment)

            yield {'data': images, 'labels': labels}

    def image_info_generator(self, images_info):
        images_name = list(images_info.keys())
        segment_size = 1000

        for n in range(0, len(images_name), segment_size):
            yield {k: images_info[k] for k in images_name[n:n + segment_size]}

    def get_info_for_images(self, annotations):
        images_info = {}

        for dirpath, dirnames, filenames in os.walk(annotations):
            annotation_files =\
                map(lambda filename: os.path.join(dirpath, filename),
                    filenames)

            for annotation in annotation_files:
                tree = ET.parse(annotation)
                root = tree.getroot()

                image_name = root.find('filename').text

                ann = self.get_image_info_for_one_image(root)
                if ann != []:
                    images_info[image_name] =\
                        self.get_image_info_for_one_image(root)

        return images_info

    def get_image_info_for_one_image(self, xml_tag):
        image_info = []

        for image_rect_points in xml_tag.iter('object'):
            if image_rect_points.find('name').text != 'person':
                continue

            image_rect_points = image_rect_points.find('bndbox')

            image_rect = ((int(float(image_rect_points.find('xmin').text)),
                           int(float(image_rect_points.find('ymin').text))),
                          (int(float(image_rect_points.find('xmax').text)),
                           int(float(image_rect_points.find('ymax').text))))
            image_center = (((image_rect[1][0] - image_rect[0][0]) / 2 +
                             image_rect[0][0]),
                            (((image_rect[1][1] - image_rect[0][1])) / 2 +
                             image_rect[0][1]))

            image_info.append([image_center, image_rect])

        return image_info


if __name__ == '__main__':
    with open('./config.json') as config_file:
        config = json.load(config_file)

    dp = VOCProcessing(config)
    dp.pickle_dataset()

    print('Taken time: {}'.format(str(dp.get_time())))
