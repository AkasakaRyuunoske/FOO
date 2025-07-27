from django.urls import path
from . import views
from .views import CreateRecipeView

urlpatterns = [
    path("", views.home, name="home"),
    path("recipes/", views.recipe_list, name="recipe_list"),
    path('recipes/new/', CreateRecipeView.as_view(), name='new_recipe'),
    path('random/', views.random_recipe, name='random_recipe'),
]
