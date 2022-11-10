from django.urls import path
from .views import *

urlpatterns = [
    path('', asynctrain),
    path("api/", async_view),
]