from datetime import datetime
import numpy as np

from aovek.utils.image_processing import ImageProcessing
from aovek.visualization.predict import Predict


class VideoProcessing(ImageProcessing):

    def __init__(self, config):
        super().__init__(config)

        self.predict = Predict(config)

        self.image_size = config['image_info']['image_size']

    def process_video_file(self, video_path):
        video = self.process_video(video_path)
        resized_video = self.resize_video(video_path)

        predictions = self.predict.predict_video(resized_video)

        image = self.make_image(video, predictions)

        return image

    def make_image(self, video, predictions):
        original_size = list(video.shape)[1:-1]

        predictions[:, 0] *= original_size[0]
        predictions[:, 1] *= original_size[1]
        predictions[:, 2] *= original_size[0]
        predictions[:, 3] *= original_size[1]

        image = np.amax(video, axis=0)

        return image
