from django.conf.urls import url
from django.conf.urls.static import static

from django.conf import settings

from Aovek import views

urlpatterns = [
    url(r'^$', views.upload_video),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
