from django.conf import settings
from django.db import models


class Recipe(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255, unique=True, null=False)
    Instructions = models.TextField(max_length=6000, null=False)
    picture = models.CharField(max_length=255, null=True)

    class Meta:
        verbose_name = "Recipe"
        verbose_name_plural = "Recipes"

    def __str__(self): return self.name


class Rating(models.Model):
    id = models.AutoField(primary_key=True)
    stars = models.IntegerField(null=False, unique=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,  # points at CustomUser model
        on_delete=models.CASCADE, db_column='user_id', related_name='ratings')
    recipe = models.ForeignKey(Recipe, on_delete=models.CASCADE, db_column='recipe_id', related_name='ratings')

    class Meta:
        verbose_name = "Rating"
        verbose_name_plural = "Ratings"

    def __str__(self):
        return f"{self.stars} stars by {self.user.username} on {self.recipe.name}"
