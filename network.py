from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape
from keras.preprocessing.image import ImageDataGenerator

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

        self. model = self.create_model()

    def create_model(self):
        input = Input(shape=(self.image_size, self.image_size,
                             self.color_channels))

        network = self.create_network(input)

        model = Model(input, network)
        model.compile(optimizer='adam',
                      loss=self.custom_loss,
                      metrics=['accuracy'])
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

        test_accuracy = self.model.evaluate(test_data, test_labels)
        train_accuracy = self.model.evaluate(train_data, train_labels)
        validation_accuracy =\
            self.model.evaluate(validation_data, validation_labels)

        print("Test accuracy: {}".format(test_accuracy))
        print("Train accuracy: {}".format(train_accuracy))
        print("Validation accuracy: {}".format(validation_accuracy))
