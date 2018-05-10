import xml.etree.ElementTree as ET

from aovek.preprocess.data_processing import DataProcessing


class CVPR10Processing(DataProcessing):

    def __init__(self, config):
        super().__init__(config)

    def generate_dataset(self):
        for train_data in self.get_train():
            self.update_pickle(self.train_pickle_name, train_data)

        for validation_data in self.get_validation():
            self.update_pickle(self.validation_pickle_name, validation_data)

        for test_data in self.get_test():
            self.update_pickle(self.test_pickle_name, test_data)

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
