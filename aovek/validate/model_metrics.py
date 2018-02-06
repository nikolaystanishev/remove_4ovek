from keras.callbacks import Callback

from aovek.validate.metrics import Metrics


class ModelMetrics(Callback, Metrics):

    def __init__(self, validation_data, validation_labels, network):
        super().__init__()
        self.validation_data = validation_data
        self.validation_labels = validation_labels

        self.iou_threshold = 0.5
        self.prob_threshold = 0.5

        self.image_size = 288
        self.grid_size = 9
        self.number_of_annotations = 4

        self.network = network

        self.validation_metrics = {}

    def on_epoch_end(self, epoch, logs={}):
        validation_metrics =\
            self.eval_metrics(self.validation_data[0], self.validation_labels)

        self.validation_metrics[epoch] = validation_metrics

        print('\nValidation IOU: {}, Validation Precision: {}, '
              'Validation Recall: {}, Validation F1 Score: {}'
              .format(validation_metrics['iou'],
                      validation_metrics['precision'],
                      validation_metrics['recall'],
                      validation_metrics['f1_score']))

        return validation_metrics

    def eval_model_metrics(self, images, labels):
        return self.eval_metrics(images, labels)

    def get_validation_metrics(self):
        iou = []
        precision = []
        recall = []
        f1_score = []

        for k in range(len(self.validation_metrics)):
            iou.append(self.validation_metrics[k]['iou'])
            precision.append(self.validation_metrics[k]['precision'])
            recall.append(self.validation_metrics[k]['recall'])
            f1_score.append(self.validation_metrics[k]['f1_score'])

        return {'iou': iou, 'precision': precision, 'recall': recall,
                'f1_score': f1_score}
