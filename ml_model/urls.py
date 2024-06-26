from django.urls import path
from .views import *

urlpatterns = [
    path('', delete_model),
]
