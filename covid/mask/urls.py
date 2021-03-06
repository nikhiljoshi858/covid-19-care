# Project Name:       COVID-19 Care: Face Mask and Social Distancing Detection using Deep Learning
# Author List:        Generated by Django
# Filename:           urls.py
# Functions:          No functions.
# Global Variables:   NA

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
from mask import views as mv
from account.views import login_view as lv

app_name = 'mask'

urlpatterns = [
    path('', mv.homepage_view, name='homepage'),
    path('image/', mv.image_view, name='image'),
    path('video/', mv.video_view, name='video'),
    path('webcam/', mv.webcam_view, name='webcam'),
    path('login/', lv, name='login'),
]
