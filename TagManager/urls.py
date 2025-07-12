from django.urls import path

from . import views

urlpatterns = [
    path('tags/', views.get_tags, name='get_tags'),
    path('get_recipes_by_tags/', views.get_recipes_by_tags, name='get_recipes_by_tags'),
    path("search_tags/", views.search_tags, name="search_tags"),
    path('ingredients/', views.get_ingredients, name='get_ingredients'),
]