import skvideo.io
import numpy as np
from skimage.transform import resize

from aovek.utils.image_processing import ImageProcessing


class VideoProcessing(ImageProcessing):

    def __init__(self, config):
        super().__init__(config)

    def process_video(self, video_path):
        video = skvideo.io.vread(video_path)

        return video

    def resize_video(self, video_path):
        video = skvideo.io.vread(video_path, as_grey=True)

        video = self.normalize_image(video)

        resized_video =\
            np.ndarray(shape=(0, self.image_size, self.image_size,
                              self.color_channels), dtype=np.float32)

        for frame in video:
            frame_data = np.squeeze(frame, axis=2)

            processed_frame =\
                resize(frame_data,
                       output_shape=(self.image_size, self.image_size),
                       mode='constant')

            processed_frame = np.expand_dims(processed_frame, axis=0)
            processed_frame = np.expand_dims(processed_frame, axis=3)

            resized_video = np.concatenate((resized_video, processed_frame))

        return resized_video
