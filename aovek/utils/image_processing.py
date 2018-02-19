from scipy import ndimage
from skimage.transform import resize
import numpy as np
import skvideo.io


class ImageProcessing:

    def __init__(self, config):
        self.image_size = config['image_info']['image_size']
        self.pixel_depth = config['image_info']['pixel_depth']
        self.color_channels = config['image_info']['color_channels']
        self.color_mode = config['image_info']['color_mode']
        self.normalizer = config['image_info']['normalizer']

    def process_image(self, image_file):
        image_data =\
            ndimage.imread(image_file, mode=self.color_mode).astype(float)

        if self.color_channels == 1:
            image_data = np.expand_dims(image_data, axis=2)

        original_size = np.squeeze(image_data).shape[::-1]

        if image_data.shape != (self.image_size, self.image_size,
                                self.color_channels):
            image = resize(image_data,
                           output_shape=(self.image_size, self.image_size),
                           mode='constant')
        else:
            image = image_data

        image = np.expand_dims(image, axis=0)

        image = self.normalize_image(image)

        return (image, original_size)

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

    def normalize_image(self, image):
        if self.normalizer == '[0, 255]':
            image = self.normalize_image_without_normalization(image)
        elif self.normalizer == '[0, 1]':
            image = self.normalize_image_from_0_to_1(image)
        elif self.normalizer == '[-1, 1]':
            image = self.normalize_image_from_minus1_to_1(image)

        return image

    def normalize_image_from_minus1_to_1(self, image):
        normalized_image = (image - (self.pixel_depth / 2)) / self.pixel_depth

        return normalized_image

    def normalize_image_from_0_to_1(self, image):
        normalized_image = image / self.pixel_depth

        return normalized_image

    def normalize_image_without_normalization(self, image):
        normalized_image = image

        return normalized_image
