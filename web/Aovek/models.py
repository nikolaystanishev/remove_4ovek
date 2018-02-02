from django.db import models
from django.db.models.signals import pre_delete
from django.dispatch.dispatcher import receiver


class Video(models.Model):
    video = models.FileField()


@receiver(pre_delete, sender=Video)
def video_delete(sender, instance, **kwargs):
    instance.video.delete(save=False)
