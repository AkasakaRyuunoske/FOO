from django.db import models
from RecipeManager.models import Recipe

class Tag(models.Model):
    name = models.CharField(max_length=255, unique=True, null=False)
    description = models.CharField(max_length=255, null=True, blank=True)
    icon = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return self.name


class RecipeTag(models.Model):
    tag = models.ForeignKey(Tag, on_delete=models.CASCADE)
    recipe = models.ForeignKey(Recipe, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.recipe} - {self.tag}"