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

        image = self.make_image(video, predictions)

        return image

    def make_image(self, video, predictions):
        original_size = list(video.shape)[1:-1]

        predictions[:, :, 0] *= original_size[1] * self.left_offset
        predictions[:, :, 1] *= original_size[0] * self.up_offset
        predictions[:, :, 2] *= original_size[1] * self.right_offset
        predictions[:, :, 3] *= original_size[0] * self.down_offset

        image = np.amax(video, axis=0)

        return image
