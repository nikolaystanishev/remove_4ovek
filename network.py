from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import model_from_json
import json
import numpy as np

from metrics import precision, recall, fmeasure

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


class YOLO:
    """
        Class for YOLO Convolutional neural network
    """

    def __init__(self, config):
        self.image_size = config['image_info']['image_size']
        self.color_channels = config['image_info']['color_channels']

        self.grid_size = config['label_info']['grid_size']
        self.number_of_classes = config['label_info']['number_of_classes']
        self.number_of_annotations =\
            config['label_info']['number_of_annotations']

        self.model = self.create_model()

        self.metrics = None
        self.model_structure = None

        self.model_binary_data_file =\
            config['network']['model_binary_data_file']
        self.model_json_structure_file =\
            config['network']['json_model_structure']

    def create_model(self):
        input = Input(shape=(self.image_size, self.image_size,
                             self.color_channels))

        network = self.create_network(input)

        model = Model(input, network)
        model.compile(optimizer='adam',
                      loss=self.custom_loss,
                      metrics=['accuracy', precision, recall, fmeasure])
        model.summary()
        return model

    def create_network(self, input):
        network = Conv2D(filters=16,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         name='conv_1')(input)
        network = MaxPooling2D(pool_size=(2, 2),
                               name='pool_1')(network)

        network = Conv2D(filters=32,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         name='conv_2')(network)
        network = MaxPooling2D(pool_size=(2, 2),
                               name='pool_2')(network)

        network = Conv2D(filters=64,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         name='conv_3')(network)
        network = MaxPooling2D(pool_size=(2, 2),
                               name='pool_3')(network)

        network = Conv2D(filters=128,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         name='conv_4')(network)
        network = MaxPooling2D(pool_size=(2, 2),
                               name='pool_4')(network)

        # network = Conv2D(filters=256,
        #                  kernel_size=(3, 3),
        #                  strides=(1, 1),
        #                  padding='same',
        #                  activation='relu',
        #                  name='conv_5')(network)
        # network = MaxPooling2D(pool_size=(2, 2),
        #                        name='pool_5')(network)

        # network = Conv2D(filters=512,
        #                  kernel_size=(3, 3),
        #                  strides=(1, 1),
        #                  padding='same',
        #                  activation='relu',
        #                  name='conv_6')(network)
        # network = MaxPooling2D(pool_size=(2, 2),
        #                        name='pool_6')(network)

        network = Conv2D(filters=1024,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         name='conv_7')(network)

        network = Conv2D(filters=1024,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         name='conv_8')(network)

        network = Conv2D(filters=(self.number_of_annotations + 1 +
                                  self.number_of_classes),
                         kernel_size=(1, 1),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         name='conv_9')(network)

        network = Reshape((self.grid_size,
                           self.grid_size,
                           (self.number_of_annotations + 1 +
                            self.number_of_classes)))(network)

        return network

    def train(self, train_data, train_labels, validation_data,
              validation_labels, test_data, test_labels):
        gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08,
                                 shear_range=0.3, height_shift_range=0.08,
                                 zoom_range=0.08)

        test_gen = ImageDataGenerator()

        train_generator = gen.flow(train_data, train_labels, batch_size=64)
        test_generator = test_gen.flow(validation_data, validation_labels,
                                       batch_size=64)

        self.model.fit_generator(train_generator, steps_per_epoch=4800 // 64,
                                 epochs=5, validation_data=test_generator,
                                 validation_steps=320 // 64)

    def custom_loss(self, y_true, y_pred):
        loss = np.sum(np.power((y_true - y_pred), 2))

        return loss

    def predict(self, image):
        predict = self.model.predict(image)

        return predict

    def save_model(self):
        self.model.save(self.model_binary_data_file)

    def load_model(self):
        custom_objects = self.get_custom_objects()
        self.model = load_model(self.model_binary_data_file,
                                custom_objects=custom_objects)

    def save_json_model_structure(self):
        json_model_structure = self.model.to_json()

        with open(self.model_json_structure_file, 'w') as f:
            json.dump(json_model_structure, f)

    def load_model_from_json_structure(self):
        custom_objects = self.get_custom_objects()
        self.model = model_from_json(self.model_json_structure_file,
                                     custom_objects=custom_objects)

    def get_custom_objects(self):
        custom_objects = {"precision": precision, "recall": recall,
                          "fmeasure": fmeasure,
                          "custom_loss": self.custom_loss}

        return custom_objects

    def summary(self, train_data, train_labels, validation_data,
                validation_labels, test_data, test_labels):
        self.metrics =\
            self.genarate_metrics(train_data, train_labels, validation_data,
                                  validation_labels, test_data, test_labels)
        self.model_structure = self.genarate_model_structure()

    def genarate_metrics(self, train_data, train_labels, validation_data,
                         validation_labels, test_data, test_labels):
        metrics =\
            self.get_metrics_values(train_data, train_labels, validation_data,
                                    validation_labels, test_data, test_labels)

        return metrics

    def genarate_model_structure(self):
        model_structure = []
        self.model.summary(print_fn=lambda row: model_structure.append(row))
        model_structure = '\n'.join(model_structure)

        return model_structure

    def get_metrics_values(self, train_data, train_labels, validation_data,
                           validation_labels, test_data, test_labels):
        test_metrics = self.model.evaluate(test_data, test_labels)
        train_metrics = self.model.evaluate(train_data, train_labels)
        validation_metrics =\
            self.model.evaluate(validation_data, validation_labels)

        loss = {'test_loss': test_metrics[0],
                'train_loss': train_metrics[0],
                'validation_loss': validation_metrics[0]}

        accuracy = {'test_accuracy': test_metrics[1],
                    'train_accuracy': train_metrics[1],
                    'validation_accuracy': validation_metrics[1]}

        precision = {'test_precision': test_metrics[2],
                     'train_precision': train_metrics[2],
                     'validation_precision': validation_metrics[2]}

        recall = {'test_recall': test_metrics[3],
                  'train_recall': train_metrics[3],
                  'validation_recall': validation_metrics[3]}

        f1_score = {'test_f1_score': test_metrics[4],
                    'train_f1_score': train_metrics[4],
                    'validation_f1_score': validation_metrics[4]}

        return {'loss': loss, 'accuracy': accuracy, 'precision': precision,
                'recall': recall, 'f1_score': f1_score}

    def get_metrics(self):
        return self.metrics

    def get_model_structure(self):
        return self.model_structure
