from django.shortcuts import render

from IngredientManager.models import Ingredient


# Create your views here.
def search_ingredients(request):
    search_term = request.GET.get("search_term", "").strip().lower()

    ingredients = []

    for ingredient in Ingredient.objects.all():
        ingredient_name = ingredient.name.strip().lower()

        # Se è vuoto, mostra tutti i tag. Altrimenti, solo quelli che contengono il termine
        if not search_term or search_term in ingredient_name:
            ingredients.append(ingredient)

    return render(request, "components/ingredients_list.html", {
        "ingredients": ingredients
    })