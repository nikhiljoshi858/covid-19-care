"""covid URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
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
from django.urls import path, include
from social import views as sv
from account.views import login_view as lv

app_name = 'social'

urlpatterns = [
    path('', sv.homepage_view, name='social_homepage'),
    path('image/',sv.image_view, name='image'),
    path('video/',sv.video_view, name='video'),
    path('webcam/',sv.webcam_view, name='webcam'),
    path('login/', lv, name='login'),
]