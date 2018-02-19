from django.conf import settings

import json

from aovek.video.video_processing import VideoProcessing

with open(settings.CONFIG_FILE) as c_f:
    config = json.load(c_f)

video_processing = VideoProcessing(config)
