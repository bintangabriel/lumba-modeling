from django.urls import path
from .views import *

urlpatterns = [
    path('', object_segmentation_inference),
]
