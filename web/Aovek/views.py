from django.shortcuts import render
from django.core.files.uploadedfile import InMemoryUploadedFile

from .forms import VideoForm
from .video_processing import video_processing

import tensorflow as tf
from PIL import Image
from io import BytesIO

graph = tf.get_default_graph()


def home(request):
    return render(request, 'base.html')


def about(request):
    return render(request, 'about.html')


def make_photo(request):
    if request.method == 'GET':
        form = VideoForm()
        objects = {'form': form, 'image': None}
    elif request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save()

        with graph.as_default():
            image = video_processing.process_video_file(video.video.path)

        image_filename = video.video.name.rsplit('.', 1)[0] + '.png'
        image = Image.fromarray(image)
        tempfile_io = BytesIO()
        image.save(tempfile_io, format='PNG')
        image_file = InMemoryUploadedFile(tempfile_io, None, image_filename,
                                          'image/png',
                                          tempfile_io.getbuffer().nbytes, None)
        video.image.save(image_filename, image_file)

        form = VideoForm()
        objects = {'form': form, 'image': video.image.url}

    return render(request, 'make_photo.html', objects)
