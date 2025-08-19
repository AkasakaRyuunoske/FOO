from django.db import models


class Ingredient(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255, null=False, unique=True)
    description = models.TextField(null=True, blank=True)

    class Meta:
        verbose_name = "Ingredient"
        verbose_name_plural = "Ingredients"

    def __str__(self): return self.name