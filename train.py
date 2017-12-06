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

        self.train_data = None
        self.train_labels = None
        self.validation_data = None
        self.validation_labels = None
        self.test_data = None
        self.test_labels = None

        self.results_file_name = config['network']['results_file']

    def train(self):
        self.network.train(self.train_data, self.train_labels,
                           self.validation_data, self.validation_labels,
                           self.test_data, self.test_labels)

    def summary(self):
        self.network.summary(self.train_data, self.train_labels,
                             self.validation_data, self.validation_labels,
                             self.test_data, self.test_labels)

        metrics = self.network.get_metrics()
        model_structure = self.network.get_model_structure()

        self.print_model_structure(model_structure)
        self.print_metrics(metrics)

    def summary_to_file(self):
        log_text = self.create_log_text()

        with open(self.results_file_name, 'a') as f:
            f.write(log_text)

    def create_log_text(self):
        log_text = ''

        model_structure = self.network.get_model_structure()
        metrics = self.network.get_metrics()

        log_text += model_structure
        log_text += """
        Test Metrics:
            Loss: {}
            Accuracy: {}
            Precision: {}
            Recall: {}
            F1 Score: {}
        ____________________________________________________________
        Train Metrics:
            Loss: {}
            Accuracy: {}
            Precision: {}
            Recall: {}
            F1 Score: {}
        ____________________________________________________________
        Validation Metrics:
            Loss: {}
            Accuracy: {}
            Precision: {}
            Recall: {}
            F1 Score: {}
        ____________________________________________________________
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                   metrics['f1_score']['validation_f1_score'])

        return log_text

    def load_data(self):
        train_pickle_name = self.pickles_name['train']
        validation_pickle_name = self.pickles_name['validation']
        test_pickle_name = self.pickles_name['test']

        self.train_data, self.train_labels =\
            self.get_data_from_pickle(train_pickle_name)
        self.validation_data, self.validation_labels =\
            self.get_data_from_pickle(validation_pickle_name)
        self.test_data, self.test_labels =\
            self.get_data_from_pickle(test_pickle_name)

    def get_data_from_pickle(self, pickle_name):
        with open(pickle_name, 'rb') as dsp:
            dataset = pickle.load(dsp)
            data = dataset['data']
            labels = dataset['labels']
            del dataset
        return data, labels

    def print_model_structure(model_structure):
        print('===================================')
        print(model_structure)
        print('===================================')

    def print_metrics(metrics):
        print('===================================')
        print('Loss: {}'.format(metrics['loss']))
        print('Accuracy: {}'.format(metrics['accuracy']))
        print('Precision: {}'.format(metrics['precision']))
        print('Recall: {}'.format(metrics['recall']))
        print('F1 Score: {}'.format(metrics['f1_score']))
        print('===================================')


if __name__ == '__main__':
    with open('./config.json') as config_file:
        config = json.load(config_file)

    train = Train(config)
    train.train()
    train.summary()
    train.summary_to_file()
