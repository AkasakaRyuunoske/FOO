from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
<<<<<<< HEAD
=======
    path("recipes/", views.recipe_list, name="recipe_list"),
    path("recipe_details/", views.recipe_details, name='recipe_details'),
>>>>>>> 2ab48244d766e93b3110c4574d70ddd3bd1c973f
]