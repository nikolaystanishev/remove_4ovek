from django.conf import settings

import json

from aovek.video.video_to_image import VideoToImage

with open(settings.CONFIG_FILE) as c_f:
    config = json.load(c_f)

video_processing = VideoToImage(config)
