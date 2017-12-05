import pickle
import json

from network import YOLO


class Train:
    """
        Class for training YOLO network
    """

    def __init__(self, config):
        self.pickles_name = config['dataset']['pickle_name']

        self.network = YOLO(config)

    def train(self):
        train_data, train_labels, validation_data, validation_labels,\
            test_data, test_labels = self.load_data()

        self.network.train(train_data, train_labels, validation_data,
                           validation_labels, test_data, test_labels)

    def load_data(self):
        train_pickle_name = self.pickles_name['train']
        validation_pickle_name = self.pickles_name['validation']
        test_pickle_name = self.pickles_name['test']

        train_data, train_labels = self.get_data_from_pickle(train_pickle_name)
        validation_data, validation_labels =\
            self.get_data_from_pickle(validation_pickle_name)
        test_data, test_labels = self.get_data_from_pickle(test_pickle_name)

        return train_data, train_labels, validation_data, validation_labels,\
            test_data, test_labels

    def get_data_from_pickle(self, pickle_name):
        with open(pickle_name, 'rb') as dsp:
            dataset = pickle.load(dsp)
            data = dataset['data']
            labels = dataset['labels']
            del dataset
        return data, labels


if __name__ == '__main__':
    with open('./config.json') as config_file:
        config = json.load(config_file)

    train = Train(config)
    train.train()
