from django.shortcuts import render

from .forms import VideoForm

from aovek.video.video_processing import VideoProcessing

video_processing = VideoProcessing()


def upload_video(request):
    if request.method == 'GET':
        form = VideoForm()
    elif request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save()
        video_processing.process_video_file(video.video.path)
        form = VideoForm()
    return render(request, 'home.html', {'form': form})
