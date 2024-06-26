"""modeling URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from asynctrain import urls as train_urls
from gpu import urls as gpu_urls
from inference import urls as inference_urls
from ml_model import urls as ml_model_urls

urlpatterns = [
    path('admin/', admin.site.urls),
    path('train/', include(train_urls)),
    path('gpu/', include(gpu_urls)),
    path('inference/', include(inference_urls)),
    path('delete-model/', include(ml_model_urls))
]
