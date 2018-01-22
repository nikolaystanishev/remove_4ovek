import numpy as np
import pickle
import json
import keras.backend as K

from network import YOLO


def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    '''Calculates the F score, the weighted harmonic mean of precision and recall.

    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    '''Calculates the f-measure, the harmonic mean of precision and recall.
    '''
    return fbeta_score(y_true, y_pred, beta=1)


class EvalMetrics:

    def __init__(self, config):
        self.min_probability = config['network']['predict']['min_probability']
        self.iou_threshold = config['network']['predict']['min_probability']
        self.prob_threshold = config['network']['predict']['prob_threshold']

        self.image_size = config['image_info']['image_size']
        self.grid_size = config['label_info']['grid_size']
        self.number_of_annotations =\
            config['label_info']['number_of_annotations']
        self.batch_size = config['network']['train']['batch_size']

        self.avg_iou = 0
        self.correct = 0
        self.proposals = 0
        self.total = 0

        self.precision = 0
        self.recall = 0

        self.network = YOLO(config)

        self.train_pickle_name = config['dataset']['pickle_name']['train']
        self.validation_pickle_name =\
            config['dataset']['pickle_name']['validation']
        self.test_pickle_name = config['dataset']['pickle_name']['test']

    def eval_pickles_metrics(self):
        train_data, train_labels, validation_data, validation_labels,\
            test_data, test_labels = self.load_data()

        avg_iou, precision, recall, f1_score =\
            self.eval_pickle_metrics(train_data, train_labels)

        print('Train:')
        print('IOU: {}, Precision: {}, Recall: {}, F1 Score: {}'
              .format(avg_iou, precision, recall, f1_score))

        avg_iou, precision, recall, f1_score =\
            self.eval_pickle_metrics(validation_data, validation_labels)

        print('Validation:')
        print('IOU: {}, Precision: {}, Recall: {}, F1 Score: {}'
              .format(avg_iou, precision, recall, f1_score))

        avg_iou, precision, recall, f1_score =\
            self.eval_pickle_metrics(test_data, test_labels)

        print('Test:')
        print('IOU: {}, Precision: {}, Recall: {}, F1 Score: {}'
              .format(avg_iou, precision, recall, f1_score))

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

    def eval_pickle_metrics(self, data, labels):
        labels =\
            np.reshape(labels, (-1, self.grid_size ** 2,
                                (self.number_of_annotations + 1)))

        prediction = self.network.predict(data)
        del data
        prediction =\
            np.reshape(prediction, (-1, self.grid_size ** 2,
                                    (self.number_of_annotations + 1)))

        labels_corners = self.get_corners_from_labels(labels)
        prediction_corners = self.get_corners_from_labels(prediction)

        iou = np.ndarray(shape=(0, self.grid_size ** 2, self.grid_size ** 2),
                         dtype=np.float32)

        for num in range(labels_corners.shape[0]):
            iou_batch = np.ndarray(shape=(0, self.grid_size ** 2),
                                   dtype=np.float32)
            for t_b in range(self.grid_size ** 2):
                label_corners = np.full([self.grid_size ** 2,
                                         (self.number_of_annotations + 1)],
                                        labels_corners[num][t_b])

                iou_t_b = self.boxes_iou(label_corners,
                                         prediction_corners[num])
                iou_t_b = np.expand_dims(iou_t_b, axis=0)

                iou_batch = np.concatenate((iou_batch, iou_t_b))

            iou_batch = np.expand_dims(iou_batch, axis=0)
            iou = np.concatenate((iou, iou_batch))

        iou = np.amax(iou, axis=2)

        best_iou = iou[np.where(iou > self.iou_threshold)]

        predicted_correct = best_iou.shape[0]
        predicted_proposals = prediction_corners[
            np.where(prediction_corners[:, :, 4] > self.prob_threshold)]\
            .shape[0]
        total_true = labels[np.where(labels[:, :, 4] == 1)].shape[0]
        best_iou = np.sum(best_iou)

        avg_iou = best_iou / predicted_correct
        precision = predicted_correct / predicted_proposals
        recall = predicted_correct / total_true
        f1_score = (2 * precision * recall) / (precision + recall)

        return avg_iou, precision, recall, f1_score

    def get_corners_from_labels(self, labels):
        corners = labels

        corners[:, :, 0] =\
            (labels[:, :, 0] - (labels[:, :, 2] / 2)) * self.image_size
        corners[:, :, 1] =\
            (labels[:, :, 1] - (labels[:, :, 3] / 2)) * self.image_size
        corners[:, :, 2] =\
            (labels[:, :, 0] + (labels[:, :, 2] / 2)) * self.image_size
        corners[:, :, 3] =\
            (labels[:, :, 1] + (labels[:, :, 3] / 2)) * self.image_size

        return corners

    def boxes_iou(self, box1, box2):
        xA = np.maximum(box1[:, 0], box2[:, 0])
        yA = np.maximum(box1[:, 1], box2[:, 1])
        xB = np.minimum(box1[:, 2], box2[:, 2])
        yB = np.minimum(box1[:, 3], box2[:, 3])

        interArea = (xB - xA) * (yB - yA)

        box1Area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        box2Area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        iou = interArea / (box1Area + box2Area - interArea)

        return iou


if __name__ == '__main__':
    with open('./config.json') as config_file:
        config = json.load(config_file)

    eval_metrics = EvalMetrics(config)
    eval_metrics.eval_pickles_metrics()
