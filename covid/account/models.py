from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from datetime import datetime
import pytz

IST = pytz.timezone('Asia/Kolkata')

# Create your models here.

class MyAccountManager(BaseUserManager):
    # All required fields are passed on the arguments
    def create_user(self, email, username, password=None):
        if not email:
            raise ValueError('Users must have an email ID')
        if not username:
            raise ValueError('Users must have a username')
        user = self.model(
            email = self.normalize_email(email),
            username = username
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, username, password):
        user = self.create_user(
            email = self.normalize_email(email),
            username = username,
            password=password
        )
        user.is_admin = True
        user.is_superuser = True
        user.is_staff = True
        user.save(using=self._db)
        return user

        
class Account(AbstractBaseUser):
    email = models.EmailField(verbose_name='email', max_length=60, unique=True)
    username = models.CharField(max_length=30, unique=True)
    date_joined = models.DateTimeField(verbose_name='date joined', auto_now_add=True)
    last_login = models.DateTimeField(verbose_name='last login', auto_now=True)
    is_admin = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)
    
    objects = MyAccountManager()

    # User logs in with this field
    USERNAME_FIELD = 'email'
    # Required fields
    REQUIRED_FIELDS = ['username']

    def __str__(self):
        return self.email
    
    def has_perm(self, perm, obj=None):
        return self.is_admin

    def has_module_perms(self, app_label):
        return True


class Previous_Mask(models.Model):
    timestamp = models.DateTimeField(default=datetime.now(IST))
    result = models.CharField(max_length=20)
    category = models.CharField(max_length=10, default=1)
    location = models.CharField(max_length=500, default=1)

    def __str__(self):
        return str(self.id)

    class Meta:
        verbose_name_plural = 'Previous Mask Records'


class Previous_Social(models.Model):
    timestamp = models.DateTimeField(default=datetime.now(IST))
    result = models.IntegerField()
    category = models.CharField(max_length=20, default=1)
    location = models.CharField(max_length=500, default=1)

    def __str__(self):
        return str(self.id)

    class Meta:
        verbose_name_plural = 'Previous Social Distancing Records'



