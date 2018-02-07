import pickle


class DataLoading:

    def __init__(self, config):
        self.train_pickle_name = config['dataset']['pickle_name']['train']
        self.validation_pickle_name =\
            config['dataset']['pickle_name']['validation']
        self.test_pickle_name = config['dataset']['pickle_name']['test']

        self.train_data = None
        self.train_labels = None
        self.validation_data = None
        self.validation_labels = None
        self.test_data = None
        self.test_labels = None

    def load_data(self):
        self.train_data, self.train_labels =\
            self.get_data_from_pickle(self.train_pickle_name)
        self.validation_data, self.validation_labels =\
            self.get_data_from_pickle(self.validation_pickle_name)
        self.test_data, self.test_labels =\
            self.get_data_from_pickle(self.test_pickle_name)

    def get_data_from_pickle(self, pickle_name):
        with open(pickle_name, 'rb') as dsp:
            dataset = pickle.load(dsp)
            data = dataset['data']
            labels = dataset['labels']
            del dataset
        return data, labels
