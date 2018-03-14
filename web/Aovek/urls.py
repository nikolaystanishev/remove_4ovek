from django.conf.urls import url
from django.conf.urls.static import static

from django.conf import settings

from Aovek import views

urlpatterns = [
    url(r'^$', views.home),
    url(r'^make_photo$', views.make_photo),
    url(r'^about$', views.about),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
