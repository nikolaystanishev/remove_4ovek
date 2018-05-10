import pickle
import numpy as np


class DataLoading:

    def __init__(self, config):
        self.train_pickle_names = []
        self.validation_pickle_names = []
        self.test_pickle_names = []

        self.train_data = None
        self.train_labels = None
        self.validation_data = None
        self.validation_labels = None
        self.test_data = None
        self.test_labels = None

    def get_datasets(self, config):
        for dataset in config['dataset']['dataset']:
            self.train_pickle_names.append(
                config['dataset'][dataset]['pickle_name']['train'])
            self.validation_pickle_names.append(
                config['dataset'][dataset]['pickle_name']['validation'])
            self.test_pickle_names.append(
                config['dataset'][dataset]['pickle_name']['test'])

    def load_data(self):
        self.train_data, self.train_labels =\
            self.load_pickle(self.train_pickle_names)
        self.validation_data, self.validation_labels =\
            self.load_pickle(self.validation_pickle_names)
        self.test_data, self.test_labels =\
            self.load_pickle(self.test_pickle_names)

    def load_pickle(self, pickle_names):
        data = np.array([])
        labels = np.array([])

        for pickle_name in pickle_names:
            new_data, new_labels = self.get_data_from_pickle(pickle_name)

            data =\
                np.vstack((data, new_data))
            labels =\
                np.vstack((labels, new_labels))

        return data, labels

    def get_data_from_pickle(self, pickle_name):
        with open(pickle_name, 'rb') as dsp:
            dataset = pickle.load(dsp)
            data = dataset['data']
            labels = dataset['labels']
            del dataset
        return data, labels
