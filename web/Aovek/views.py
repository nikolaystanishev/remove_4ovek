from django.shortcuts import render

import skvideo.io

from .forms import VideoForm


def process_file(file_path):
    video = skvideo.io.vread(file_path)
    print(video.shape)


def upload_video(request):
    if request.method == 'GET':
        form = VideoForm()
    elif request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save()
        process_file(video.video.path)
        video.delete()
        form = VideoForm()
    return render(request, 'home.html', {'form': form})
