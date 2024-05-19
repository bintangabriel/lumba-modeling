from django.urls import path
from .views import *

urlpatterns = [
    path('', GPUChecker.as_view())
]
