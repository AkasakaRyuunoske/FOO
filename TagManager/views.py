import json
from time import sleep

from django.http import HttpResponse
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

    # ritorno il componente che itera sugli tag e per ogni tag crea una card
    return render(request, "components/recipe_cards.html", {"tags": tags})


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
