from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("recipes/", views.recipe_list, name="recipe_list"),
    path("recipe_details/", views.recipe_details, name='recipe_details'),
]