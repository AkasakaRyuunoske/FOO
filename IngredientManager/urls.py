from django.urls import path

from . import views

urlpatterns = [
    path("search_ingredients/", views.search_ingredients, name="search_ingredients"),
]