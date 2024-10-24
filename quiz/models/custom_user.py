from django.db import models 
from django.contrib.auth.models import AbstractUser


class CustomUser(AbstractUser):
    ROLE = (
        ('student', 'student'),
        ('teacher', 'teacher'),
    )
    role = models.CharField(max_length=15, default='student', choices=ROLE)


    class Meta:
        verbose_name = 'Tài khoản'
        verbose_name_plural = 'Tài khoản'


    def __str__(self):
        return f"{self.id} - {self.username} - {self.role}"
    

    def save(self, *args, **kwargs):
        if self.role == 'teacher':
            self.is_active = True 
            self.is_staff = True
            self.is_superuser = True 
        super(CustomUser, self).save(*args, **kwargs)
