import pickle
import json
from datetime import datetime
from prettytable import PrettyTable

from network import YOLO


class Train:
    """
        Class for training YOLO network
    """

    def __init__(self, config):
        start_time = datetime.now()

        self.network = YOLO(config)

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

        self.load_data()

        self.results_file_name = config['network']['results_file']

        self.train_time = None
        self.full_time = None

        self.train()

        end_time = datetime.now()
        self.full_time = end_time - start_time

        self.summary()
        self.summary_to_file()

    def train(self):
        start_time = datetime.now()
        self.network.train(self.train_data, self.train_labels,
                           self.validation_data, self.validation_labels,
                           self.test_data, self.test_labels)
        end_time = datetime.now()
        self.train_time = end_time - start_time

        self.network.save_model()
        self.network.save_json_model_structure()

    def summary(self):
        self.network.summary(self.train_data, self.train_labels,
                             self.validation_data, self.validation_labels,
                             self.test_data, self.test_labels)

        metrics = self.network.get_metrics()
        model_structure = self.network.get_model_structure()
        model_history = self.network.get_model_history()

        self.print_model_structure(model_structure)
        self.print_model_history(model_history)
        self.print_metrics(metrics)

    def summary_to_file(self):
        log_text = self.create_log_text()

        with open(self.results_file_name, 'a') as f:
            f.write(log_text)

    def create_log_text(self):
        log_text = ''

        model_structure = self.network.get_model_structure()
        metrics = self.network.get_metrics()
        model_history = self.network.get_model_history()

        log_text += model_structure
        log_text += '\n'

        temp_model_history = PrettyTable()
        temp_model_history.add_column(
            'Epoch', ['Epoch ' + str(el)
                      for el in range(1, len(model_history['acc']) + 1)])
        for k, v in sorted(model_history.items()):
            temp_model_history.add_column(k, v)
        log_text += str(temp_model_history)

        log_text += """
Test Metrics:
    Loss       : {}
    Accuracy   : {}
    Precision  : {}
    Recall     : {}
    F1 Score   : {}
_________________________________________________________________
Train Metrics:
    Loss       : {}
    Accuracy   : {}
    Precision  : {}
    Recall     : {}
    F1 Score   : {}
_________________________________________________________________
Validation Metrics:
    Loss       : {}
    Accuracy   : {}
    Precision  : {}
    Recall     : {}
    F1 Score   : {}
_________________________________________________________________
Time:
    Train Time : {}
    Full Time  : {}
_________________________________________________________________
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""".format(metrics['loss']['test_loss'],
           metrics['accuracy']['test_accuracy'],
           metrics['precision']['test_precision'],
           metrics['recall']['test_recall'],
           metrics['f1_score']['test_f1_score'],
           metrics['loss']['train_loss'],
           metrics['accuracy']['train_accuracy'],
           metrics['precision']['train_precision'],
           metrics['recall']['train_recall'],
           metrics['f1_score']['train_f1_score'],
           metrics['loss']['validation_loss'],
           metrics['accuracy']['validation_accuracy'],
           metrics['precision']['validation_precision'],
           metrics['recall']['validation_recall'],
           metrics['f1_score']['validation_f1_score'],
           self.train_time,
           self.full_time)

        return log_text

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

    def print_model_structure(self, model_structure):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(model_structure)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    def print_model_history(self, model_history):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        for k, v in sorted(model_history.items()):
            print('{}: {}'.format(k, v))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    def print_metrics(self, metrics):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Loss: {}'.format(metrics['loss']))
        print('Accuracy: {}'.format(metrics['accuracy']))
        print('Precision: {}'.format(metrics['precision']))
        print('Recall: {}'.format(metrics['recall']))
        print('F1 Score: {}'.format(metrics['f1_score']))
        print('Train Time: {}'.format(self.train_time))
        print('Full Time: {}'.format(self.full_time))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


if __name__ == '__main__':
    with open('./config.json') as config_file:
        config = json.load(config_file)

    train = Train(config)
