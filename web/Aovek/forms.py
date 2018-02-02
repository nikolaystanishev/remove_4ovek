from django import forms

from .models import Video


class VideoForm(forms.ModelForm):
    class Meta:
        model = Video
        fields = ['video']

    def __init__(self, *args, **kwargs):
        super(VideoForm, self).__init__(*args, **kwargs)
        self.fields['video'].label = 'Video'
        self.fields['video'].widget.attrs['placeholder'] = 'Video'
        self.fields['video'].widget.attrs['class'] = 'form-control'
