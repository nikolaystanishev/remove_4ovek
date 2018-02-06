import pickle
import json

from aovek.validate.metrics import Metrics
from aovek.network.network import YOLO


class EvalMetrics(Metrics):

    def __init__(self, config):
        super().__init__(config)

        self.network = YOLO(config)
        self.network.load_model()

        self.train_pickle_name = config['dataset']['pickle_name']['train']
        self.validation_pickle_name =\
            config['dataset']['pickle_name']['validation']
        self.test_pickle_name = config['dataset']['pickle_name']['test']

    def eval_pickles_metrics(self):
        train_data, train_labels, validation_data, validation_labels,\
            test_data, test_labels = self.load_data()

        print('Train:')
        self.eval_dataset_metrics(train_data, train_labels)

        print('Validation:')
        self.eval_dataset_metrics(validation_data, validation_labels)

        print('Test:')
        self.eval_dataset_metrics(test_data, test_labels)

    def load_data(self):
        train_data, train_labels =\
            self.get_data_from_pickle(self.train_pickle_name)
        validation_data, validation_labels =\
            self.get_data_from_pickle(self.validation_pickle_name)
        test_data, test_labels =\
            self.get_data_from_pickle(self.test_pickle_name)

        return train_data, train_labels, validation_data, validation_labels,\
            test_data, test_labels

    def get_data_from_pickle(self, pickle_name):
        with open(pickle_name, 'rb') as dsp:
            dataset = pickle.load(dsp)
            data = dataset['data']
            labels = dataset['labels']
            del dataset
        return data, labels

    def eval_dataset_metrics(self, data, labels):
        metrics = self.eval_metrics(data, labels)

        print('IOU: {}, Precision: {}, Recall: {}, F1 Score: {}'
              .format(metrics['iou'], metrics['precision'],
                      metrics['recall'], metrics['f1_score']))


if __name__ == '__main__':
    with open('./config.json') as config_file:
        config = json.load(config_file)

    eval_metrics = EvalMetrics(config)
    eval_metrics.eval_pickles_metrics()
