import skvideo.io


class VideoProcessing:
    def __init__(self):
        pass

    def process_video_file(self, file_path):
        video = skvideo.io.vread(file_path)
        print(video.shape)
