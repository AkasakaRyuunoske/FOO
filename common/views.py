import random

from django.core.paginator import Paginator
from django.shortcuts import render

from RecipeManager.models import Recipe


def home(request):
    # Step 1: Generate and persist random IDs
    if "random_recipe_ids" not in request.session:
        all_ids = list(Recipe.objects.values_list("id", flat=True))
        random_ids = random.sample(all_ids, min(60, len(all_ids)))
        request.session["random_recipe_ids"] = random_ids

    # Step 2: Retrieve recipes from session
    recipe_ids = request.session["random_recipe_ids"]
    recipes_list = Recipe.objects.filter(id__in=recipe_ids)

    # Step 3: Pagination
    paginator = Paginator(recipes_list, 10)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    return render(request, "home.html", {"page_obj": page_obj})


def recipe_list(request):
    recipes_list = Recipe.objects.all()
    paginator = Paginator(recipes_list, 10)

    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    return render(request, "recipes/recipe_list.html", {"page_obj": page_obj})


def get_random_n_recipes(n=60):
    ids = list(Recipe.objects.values_list('id', flat=True))
    random_ids = random.sample(ids, min(len(ids), n))  # In case there are <60 recipes
    return Recipe.objects.filter(id__in=random_ids)


def discover(request):
    return render(request, "discover.html", None)


def recipe_details(request):
    return render(request, "recipe_details.html", None)
