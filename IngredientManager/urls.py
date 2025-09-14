from django.urls import path

from . import views

urlpatterns = [
    path("search_ingredients/", views.search_ingredients, name="search_ingredients"),
    path("get_recipes_by_ingredients/", views.get_recipes_by_ingredients, name="get_recipes_by_ingredients"),
]