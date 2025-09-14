from django.urls import path

from common import views

urlpatterns = [
    path("modal/<int:recipe_id>/", views.load_recipe_modal, name="load_recipe_modal"),
]

