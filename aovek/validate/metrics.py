import numpy as np


class Metrics:

    def __init__(self, config):
        self.iou_threshold = config['network']['predict']['iou_threshold']
        self.prob_threshold = config['network']['predict']['prob_threshold']

        self.image_size = config['image_info']['image_size']
        self.grid_size = config['label_info']['grid_size']
        self.number_of_annotations =\
            config['label_info']['number_of_annotations']

    def eval_metrics(self, images, labels):
        iou, gt_num, tp, fp, fn = self.get_metrics_params(images, labels)

        iou, precision, recall, f1_score =\
            self.calculate_metrics(iou, gt_num, tp, fp, fn)

        return {'iou': iou, 'precision': precision, 'recall': recall,
                'f1_score': f1_score}

    def get_metrics_params(self, images, labels):
        iou = 0
        gt_num = 0

        tp = 0
        fp = 0
        fn = 0

        preds = self.network.predict_images(images)
        preds[:, :, :4] = preds[:, :, :4] * self.image_size

        for image, label, pred in zip(images, labels, preds):
            pred = pred[~np.all(pred == 0, axis=1)]

            iou_image, gt_num_image, tp_image, fp_image, fn_image =\
                self.get_one_image_metrics_params(image, label, pred)

            iou += iou_image
            gt_num += gt_num_image

            tp += tp_image
            fp += fp_image
            fn += fn_image

        return iou, gt_num, tp, fp, fn

    def get_one_image_metrics_params(self, image, label, pred):
        label = np.reshape(label, (self.grid_size ** 2,
                                   (self.number_of_annotations + 1)))

        gt = label[np.where(label[:, 4] == 1)]
        gt = self.get_corners_from_labels(gt)

        image = np.expand_dims(image, axis=0)

        iou_image = self.get_iou_for_image(gt, pred)

        iou, gt_num, tp, fp, fn =\
            self.get_metrics_params_from_iou(iou_image, gt, pred)

        return iou, gt_num, tp, fp, fn

    def get_iou_for_image(self, gt, pred):
        iou_image = np.ndarray(shape=(0, pred.shape[0]),
                               dtype=np.float32)

        for box in gt:
            gt_box = np.full(pred.shape, box)

            iou_box = self.boxes_iou(gt_box, pred)

            iou_box = np.expand_dims(iou_box, axis=0)
            iou_image = np.concatenate((iou_image, iou_box))

        return iou_image

    def get_metrics_params_from_iou(self, iou_image, gt, pred):
        if iou_image.shape[1] != 0:
            iou_image = np.amax(iou_image, axis=1)
            iou_image = iou_image.flatten()

            gt_num = gt.shape[0]

            true_boxes =\
                iou_image[np.where(iou_image >= self.iou_threshold)]

            tp = true_boxes.shape[0]
            fp = pred.shape[0] - true_boxes.shape[0]
            fn = gt.shape[0] - true_boxes.shape[0]
            iou = np.sum(true_boxes)
        else:
            iou = 0
            gt_num = gt.shape[0]

            tp = 0
            fp = pred.shape[0]
            fn = gt.shape[0]

        return iou, gt_num, tp, fp, fn

    def calculate_metrics(self, iou, gt_num, tp, fp, fn):
        iou = self.save_div(iou, tp)

        precision = self.save_div(tp, (tp + fp))
        recall = self.save_div(tp, (tp + fn))
        f1_score = self.save_div((2 * precision * recall),
                                 (precision + recall))

        return iou, precision, recall, f1_score

    def save_div(self, num1, num2):
        try:
            return num1 / num2
        except ZeroDivisionError:
            return 0

    def get_corners_from_labels(self, labels):
        corners = np.array(labels, copy=True)

        corners[:, 0] =\
            (labels[:, 0] - (labels[:, 2] / 2)) * self.image_size
        corners[:, 1] =\
            (labels[:, 1] - (labels[:, 3] / 2)) * self.image_size
        corners[:, 2] =\
            (labels[:, 0] + (labels[:, 2] / 2)) * self.image_size
        corners[:, 3] =\
            (labels[:, 1] + (labels[:, 3] / 2)) * self.image_size

        return corners

    def boxes_iou(self, box1, box2):
        ymin_1 = np.minimum(box1[:, 0], box1[:, 2])
        xmin_1 = np.minimum(box1[:, 1], box1[:, 3])
        ymax_1 = np.maximum(box1[:, 0], box1[:, 2])
        xmax_1 = np.maximum(box1[:, 1], box1[:, 3])
        ymin_2 = np.minimum(box2[:, 0], box2[:, 2])
        xmin_2 = np.minimum(box2[:, 1], box2[:, 3])
        ymax_2 = np.maximum(box2[:, 0], box2[:, 2])
        xmax_2 = np.maximum(box2[:, 1], box2[:, 3])

        area_1 = (ymax_1 - ymin_1) * (xmax_1 - xmin_1)
        area_2 = (ymax_2 - ymin_2) * (xmax_2 - xmin_2)

        ymin_inter = np.maximum(ymin_1, ymin_2)
        xmin_inter = np.maximum(xmin_1, xmin_2)
        ymax_inter = np.minimum(ymax_1, ymax_2)
        xmax_inter = np.minimum(xmax_1, xmax_2)

        area_inter = (np.maximum(ymax_inter - ymin_inter, 0.0) *
                      np.maximum(xmax_inter - xmin_inter, 0.0))

        iou = area_inter / (area_1 + area_2 - area_inter)

        iou[np.where(area_1 < 0) or np.where(area_2 < 0)] = 0

        return iou
