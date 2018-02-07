import json

from aovek.validate.metrics import Metrics
from aovek.network.network import YOLO
from aovek.utils.data_loading import DataLoading


class EvalMetrics(Metrics, DataLoading):

    def __init__(self, config):
        Metrics.__init__(self, config)
        DataLoading.__init__(self, config)

        self.network = YOLO(config)
        self.network.load_model()

        self.load_data()

    def eval_pickles_metrics(self):
        print('Train:')
        self.eval_dataset_metrics(self.train_data, self.train_labels)

        print('Validation:')
        self.eval_dataset_metrics(self.validation_data, self.validation_labels)

        print('Test:')
        self.eval_dataset_metrics(self.test_data, self.test_labels)

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
