from django.db import models


# ID - int, autoincrement, primary key
# Name - varchar, not null, unique
# Instructions - text, not null
# Picture - varchar, nullable

class Recipe(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=255, unique=True, null=False)
    Instructions = models.TextField(max_length=6000, null=False)
    picture = models.CharField(max_length=255, null=True)

    class Meta:
        verbose_name = "Recipe"
        verbose_name_plural = "Recipes"

    def __str__(self): return self.name
