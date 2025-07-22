import random

from django.core.paginator import Paginator
from django.http import HttpResponseBadRequest
from django.shortcuts import render, redirect
from django.views import View

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
    # TODO: Qui va il modello
    recipes = Recipe.objects.all()

    # Paginazione
    page_number = request.GET.get("page", 1)
    paginator = Paginator(recipes, 12)
    page_obj = paginator.get_page(page_number)

    # Ritorna il componente che itera sulle ricette
    return render(request, "discover.html", {
        "page_obj": page_obj,
    })


class CreateRecipeView(View):
    template_name = 'new_recipe.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        # simple validation
        name = request.POST.get('name')
        desc = request.POST.get('description')
        cook_val = request.POST.get('cook_time_value')
        cook_unit = request.POST.get('cook_time_unit')
        difficulty = request.POST.get('difficulty')
        servings = request.POST.get('servings')

        if not all([name, desc, cook_val, cook_unit, difficulty, servings]):
            return HttpResponseBadRequest("Missing required fields")

        r = Recipe.objects.create(
            name=name,
            instructions=desc,
            # you'll need to handle photo upload separately,
            # e.g. request.FILES['photo'] and a proper ImageField on Recipe
        )
        # you can parse and save ingredients/tags manually here…

        return redirect('recipe_detail', pk=r.pk)
