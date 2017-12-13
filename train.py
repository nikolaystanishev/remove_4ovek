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

    def train(self):
        start_time = datetime.now()
        self.network.train(self.train_data, self.train_labels,
                           self.validation_data, self.validation_labels)
        end_time = datetime.now()
        self.train_time = end_time - start_time

        self.network.save_model()
        self.network.save_json_model_structure()

    def summary(self):
        self.network.summary(self.train_data, self.train_labels,
                             self.validation_data, self.validation_labels,
                             self.test_data, self.test_labels)

        log_text = self.create_log_text()

        with open(self.results_file_name, 'a') as f:
            f.write(log_text)

    def create_log_text(self):
        log_text = ''

        model_structure = self.network.get_model_structure()

        log_text += model_structure
        log_text += '\n'

        log_text += self.get_model_history_log()
        log_text += """
_________________________________________________________________
"""

        log_text += self.get_optimazer_log()

        log_text += """
_________________________________________________________________
"""

        log_text += self.get_metrics_log()

        log_text += self.get_time_log()

        self.print_log(log_text)

        return log_text

    def get_model_history_log(self):
        model_history = self.network.get_model_history()

        temp_model_history = PrettyTable()

        temp_model_history.add_column(
            'Epoch', ['Epoch ' + str(el)
                      for el in range(1, len(model_history['acc']) + 1)])
        for k, v in sorted(model_history.items()):
            temp_model_history.add_column(k, v)

        model_history_log = str(temp_model_history)

        return model_history_log

    def get_optimazer_log(self):
        optimizer_type = self.network.get_optimizer_type()
        optimizer_params = self.network.get_optimizer_params()

        temp_optimizer_params = PrettyTable()

        temp_optimizer_params.add_column('Optimizer', [optimizer_type])
        for k, v in sorted(optimizer_params.items()):
            temp_optimizer_params.add_column(k, [v])

        optimizer_log = str(temp_optimizer_params)

        return optimizer_log

    def get_metrics_log(self):
        metrics = self.network.get_metrics()

        temp_metrics = PrettyTable()

        temp_metrics.add_column('Metrics', ['Test Metrics', 'Train Metrics',
                                            'Validation Metrics'])

        temp_metrics.add_column('Loss',
                                [metrics['loss']['test_loss'],
                                 metrics['loss']['train_loss'],
                                 metrics['loss']['validation_loss']])
        temp_metrics.add_column('Accuracy',
                                [metrics['accuracy']['test_accuracy'],
                                 metrics['accuracy']['train_accuracy'],
                                 metrics['accuracy']['validation_accuracy']])
        temp_metrics.add_column('Precision',
                                [metrics['precision']['test_precision'],
                                 metrics['precision']['train_precision'],
                                 metrics['precision']['validation_precision']])
        temp_metrics.add_column('Recall',
                                [metrics['recall']['test_recall'],
                                 metrics['recall']['train_recall'],
                                 metrics['recall']['validation_recall']])
        temp_metrics.add_column('F1 Score',
                                [metrics['f1_score']['test_f1_score'],
                                 metrics['f1_score']['train_f1_score'],
                                 metrics['f1_score']['validation_f1_score']])

        metrics_log = str(temp_metrics)

        return metrics_log

    def get_time_log(self):
        time_log = """
_________________________________________________________________
Time:
    Train Time : {}
    Full Time  : {}
_________________________________________________________________
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""".format(self.train_time,
           self.full_time)

        return time_log

    def print_log(self, log):
        print(log)


if __name__ == '__main__':
    with open('./config.json') as config_file:
        config = json.load(config_file)

    train = Train(config)
