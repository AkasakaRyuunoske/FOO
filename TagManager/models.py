from django.db import models

class Tag(models.Model):
    name = models.CharField(max_length=255, unique=True, null=False)
    description = models.CharField(max_length=255, null=True, blank=True)
    icon = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return self.name

