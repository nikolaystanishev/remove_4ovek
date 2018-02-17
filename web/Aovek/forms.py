from django import forms

from .models import Video


class VideoForm(forms.ModelForm):
    class Meta:
        model = Video
        fields = ['video']
        labels = {'video': ''}

    def __init__(self, *args, **kwargs):
        super(VideoForm, self).__init__(*args, **kwargs)
        self.fields['video'].widget.attrs['class'] = 'custom-file-input'
        self.fields['video'].widget.attrs['id'] = 'customFile'
