# Project Name:       COVID-19 Care: Face Mask and Social Distancing Detection using Deep Learning
# Author List:        Generated by Django
# Filename:           admin.py
# Functions:          No functions. This file is used for regsitering models on the admin panel
# Global Variables:   NA

from django.contrib import admin
from account.models import *
# Register your models here.
admin.site.register(Account)
admin.site.register(Previous_Mask)
admin.site.register(Previous_Social)