from django.db import models
from RecipeManager.models import Recipe

class TagType(models.Model):
    name = models.CharField(max_length=255, unique=True, null=False)

    def __str__(self):
        return self.name

class Tag(models.Model):
    name = models.CharField(max_length=255, null=False)
    tag_type = models.ForeignKey(TagType, on_delete=models.CASCADE, related_name='tags', null=True)

    def __str__(self):
        return f"{self.name} ({self.tag_type.name})"


class RecipeTag(models.Model):
    tag = models.ForeignKey(Tag, on_delete=models.CASCADE)
    recipe = models.ForeignKey(Recipe, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.recipe} - {self.tag}"