from django.urls import path
from .views import *

urlpatterns = [
    path('', async_train_endpoint),
    path('download/', download_model)
]
