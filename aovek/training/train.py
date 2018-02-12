import json
from datetime import datetime
from prettytable import PrettyTable

from aovek.network.network import YOLO
from aovek.utils.data_loading import DataLoading


class Train(DataLoading):
    """
        Class for training YOLO network
    """

    def __init__(self, config):
        super().__init__(config)

        self.network = None

        self.results_file_name = config['network']['results_file']

        self.train_time = None
        self.dataset_loading_time = None
        self.metrics_evaluation_time = None

    def load_dataset(self):
        start_time = datetime.now()

        self.load_data()

        end_time = datetime.now()

        self.dataset_loading_time = end_time - start_time

    def train(self, config):
        start_time = datetime.now()

        self.network = YOLO(config)
        self.network.train(self.train_data, self.train_labels,
                           self.validation_data, self.validation_labels)

        end_time = datetime.now()
        self.train_time = end_time - start_time

        self.network.save_model()
        self.network.save_json_model_structure()

        start_time = datetime.now()

        self.summary()

        end_time = datetime.now()
        self.metrics_evaluation_time = end_time - start_time

        self.log()

    def summary(self):
        self.network.summary(self.train_data, self.train_labels,
                             self.validation_data, self.validation_labels,
                             self.test_data, self.test_labels)

    def log(self):
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

        log_text += self.get_batch_size_log()

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
                      for el in range(1, len(model_history['iou']) + 1)])
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

        temp_metrics.add_column('Metrics', ['Train Metrics',
                                            'Validation Metrics',
                                            'Test Metrics'])

        temp_metrics.add_column('Loss',
                                [metrics['loss']['train_loss'],
                                 metrics['loss']['validation_loss'],
                                 metrics['loss']['test_loss']])
        temp_metrics.add_column('IoU',
                                [metrics['iou']['train_iou'],
                                 metrics['iou']['validation_iou'],
                                 metrics['iou']['test_iou']])
        temp_metrics.add_column('Precision',
                                [metrics['precision']['train_precision'],
                                 metrics['precision']['validation_precision'],
                                 metrics['precision']['test_precision']])
        temp_metrics.add_column('Recall',
                                [metrics['recall']['train_recall'],
                                 metrics['recall']['validation_recall'],
                                 metrics['recall']['test_recall']])
        temp_metrics.add_column('F1 Score',
                                [metrics['f1_score']['train_f1_score'],
                                 metrics['f1_score']['validation_f1_score'],
                                 metrics['f1_score']['test_f1_score']])

        metrics_log = str(temp_metrics)

        return metrics_log

    def get_time_log(self):
        time_log = """
_________________________________________________________________
Time:
    Train Time              : {}
    Dataset Loading Time    : {}
    Metrics Evaluation Time : {}
_________________________________________________________________
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""".format(self.train_time,
           self.dataset_loading_time,
           self.metrics_evaluation_time)

        return time_log

    def get_batch_size_log(self):
        batch_size = self.network.get_batch_size()

        temp_batch_size = PrettyTable()

        temp_batch_size.add_column('Batch Size', [batch_size])

        batch_size_log = str(temp_batch_size)

        return batch_size_log

    def print_log(self, log):
        print(log)


if __name__ == '__main__':
    with open('./config.json') as config_file:
        config = json.load(config_file)

    train = Train(config)
