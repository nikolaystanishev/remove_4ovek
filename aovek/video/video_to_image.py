from datetime import datetime
import numpy as np

from aovek.utils.video_processing import VideoProcessing
from aovek.visualization.predict import Predict


class VideoToImage(VideoProcessing):

    def __init__(self, config):
        super().__init__(config)

        self.predict = Predict(config)

        self.image_size = config['image_info']['image_size']

        self.up_offset = config['video_info']['up_offset']
        self.down_offset = config['video_info']['down_offset']
        self.left_offset = config['video_info']['left_offset']
        self.right_offset = config['video_info']['right_offset']

    def process_video_file(self, video_path):
        video = self.process_video(video_path)
        resized_video = self.resize_video(video_path)

        predictions = self.predict.predict_video(resized_video)

        original_size = list(video.shape)[1:-1]

        predictions = self.predictions_to_original_size(predictions,
                                                        original_size)

        image = self.make_image(video, predictions)

        return image

    def make_image(self, video, predictions):
        image = np.zeros(video.shape[1:])

        up_border = 0
        down_border = image.shape[0]
        left_border = 0
        right_border = image.shape[1]

        for frame, prediction in zip(video, predictions):
            if np.sum(prediction[0:4]) == 0:
                continue
            for pred in prediction:
                if np.sum(pred[0:4]) == 0:
                    continue

                image[up_border:int(pred[1]), left_border:right_border] =\
                    frame[up_border:int(pred[1]), left_border:right_border]
                if int(pred[1]) > up_border:
                    up_border = int(pred[1])

                image[int(pred[3]):down_border, left_border:right_border] =\
                    frame[int(pred[3]):down_border, left_border:right_border]
                if int(pred[3]) < down_border:
                    down_border = int(pred[3])

                image[up_border:down_border, left_border:int(pred[0])] =\
                    frame[up_border:down_border, left_border:int(pred[0])]
                if int(pred[0]) > left_border:
                    left_border = int(pred[0])

                image[up_border:down_border, int(pred[2]):right_border] =\
                    frame[up_border:down_border, int(pred[2]):right_border]
                if int(pred[2]) < right_border:
                    right_border = int(pred[2])

        return image.astype('uint8')

    def predictions_to_original_size(self, predictions, original_size):
        predictions[:, :, 0] *= original_size[1] * self.left_offset
        predictions[:, :, 1] *= original_size[0] * self.up_offset
        predictions[:, :, 2] *= original_size[1] * self.right_offset
        predictions[:, :, 3] *= original_size[0] * self.down_offset

        return predictions

    def make_video_with_rectangles(self, video, predictions, video_path):
        video_with_rectangles =\
            self.draw_rectangles_in_video(video, predictions)

        self.write_video(video_with_rectangles, video_path.split('/')[-1])

    def draw_rectangles_in_video(self, video, predictions):
        rect_color = 0

        video_with_rectangles = np.array(video, copy=True)

        for frame in range(video.shape[0]):
            for pred in predictions[frame]:
                video_with_rectangles[frame][int(pred[1]):int(pred[3]),
                                             int(pred[0]):
                                             int(pred[0]) + 5] = rect_color
                video_with_rectangles[frame][int(pred[1]):int(pred[3]),
                                             int(pred[2]):
                                             int(pred[2]) + 5] = rect_color
                video_with_rectangles[frame][int(pred[1]):int(pred[1]) + 5,
                                             int(pred[0]):
                                             int(pred[2])] = rect_color
                video_with_rectangles[frame][int(pred[3]):int(pred[3]) + 5,
                                             int(pred[0]):
                                             int(pred[2])] = rect_color

        return video_with_rectangles
