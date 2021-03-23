from django.db import models

# Create your models here.
class Video(models.Model):
    video = models.FileField(upload_to='mask_videos/')
    
    #def __str__(self):
    #    return self.id
    
    class Meta:
        verbose_name_plural = 'Videos'