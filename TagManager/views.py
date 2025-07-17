from RecipeManager.models import Recipe
from django.core.paginator import Paginator
from django.shortcuts import render


def get_tags(request):
    result = ["Fast", "Slow", "Average", "Overnight",
              "Cheap", "Average", "Expensive", "Rich",
              "Vegan", "Vegetarian",
              "Boiled", "Grilled", "Backed", "Boiled",
              "Easy", "Medium", "Hard", "Chef"]

    return render(request, "components/tags_list.html", {"tags": result})


def get_recipes_by_tags(request):
    # prendo gli tag selezionati dal utente dalla query string
    tag_string = request.GET.get("tags", "")

    # pulizia degli tag
    tags = [t.strip() for t in tag_string.split(",") if t.strip()]

    # TODO: Qui va il modello
    recipes = Recipe.objects.all()

    # Paginazione
    page_number = request.GET.get("page", 1)
    paginator = Paginator(recipes, 12)
    page_obj = paginator.get_page(page_number)

    # Ritorna il componente che itera sulle ricette
    return render(request, "components/recipe_cards.html", {
        "tags": tags,
        "page_obj": page_obj,
    })


def search_tags(request):
    all_tags = [
        "Fast", "Slow", "Average", "Overnight",
        "Cheap", "Average", "Expensive", "Rich",
        "Vegan", "Vegetarian",
        "Boiled", "Grilled", "Backed", "Boiled",
        "Easy", "Medium", "Hard", "Chef"
    ]

    query = request.GET.get("q", "").strip().lower()
    filtered = []

    if query:
        filtered = [tag for tag in all_tags if query in tag.lower()]
    else:
        filtered = all_tags

    print(f"Filtered ==> {filtered}")
    return render(request, "components/tag_results.html", {"tags": filtered})


def get_ingredients(request):
    result = ["Fast", "Slow", "Average", "Overnight",
              "Cheap", "Average", "Expensive", "Rich",
              "Vegan", "Vegetarian",
              "Boiled", "Grilled", "Backed", "Boiled",
              "Easy", "Medium", "Hard", "Chef"]

    return render(request, "components/ingredients_list.html", {"ingredients": result})
