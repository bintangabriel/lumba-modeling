from django.urls import path
from .views import *

urlpatterns = [
    path('', async_train_endpoint),
    path('forecasting/', async_train_forecasting_endpoint), 
]
