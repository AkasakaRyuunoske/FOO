import random

from django.core.paginator import Paginator
from django.http import HttpResponseBadRequest
from django.shortcuts import render, redirect
from django.views import View

from RecipeManager.models import Recipe


def home(request):
    # if user realoads without page param, generate new random recipes
    if "page" not in request.GET and "random_recipe_ids" in request.session:
        del request.session["random_recipe_ids"]

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


def discover(request):
    recipes = Recipe.objects.all()

    # Paginazione
    page_number = request.GET.get("page", 1)
    paginator = Paginator(recipes, 12)
    page_obj = paginator.get_page(page_number)

    # Ritorna il componente che itera sulle ricette
    return render(request, "discover.html", {
        "page_obj": page_obj,
    })

def random_recipe(request):
    all_ids = Recipe.objects.values_list('id', flat=True)

    rand_id = random.choice(list(all_ids))
    recipe = Recipe.objects.get(pk=rand_id)

    recipes = [recipe]

    paginator = Paginator(recipes, per_page=1)
    page_obj = paginator.page(1)

    return render(request, "components/recipe_cards.html", {
        'page_obj': page_obj,
    })


class CreateRecipeView(View):
    # Template file that will be rendered when showing the form
    template_name = 'new_recipe.html'

    def get(self, request):
        """
        Handles GET requests (when user visits the page)
        Django automatically calls this method for GET requests
        """
        return self.get_recipe_form(request)

    def post(self, request):
        """
        Handles POST requests (when user submits the form)
        Django automatically calls this method for POST requests
        """
        return self.post_recipe_form(request)

    def get_recipe_form(self, request):
        """
        Shows the empty recipe creation form to the user
        This is called when someone first visits the page
        """
        return render(request, self.template_name)

    def post_recipe_form(self, request):
        """
        Processes the submitted form data and creates a new recipe
        This is called when the user clicks submit on the form
        """
        # Extract data from the submitted form
        # request.POST.get() safely gets form field values
        name = request.POST.get('name')
        desc = request.POST.get('description')
        cook_val = request.POST.get('cook_time_value')
        cook_unit = request.POST.get('cook_time_unit')
        difficulty = request.POST.get('difficulty')
        servings = request.POST.get('servings')

        # Check if all required fields have values
        # all() returns True only if all items in the list are truthy (not empty)
        if not all([name, desc, cook_val, cook_unit, difficulty, servings]):
            # Return error response if any field is missing
            return HttpResponseBadRequest("Missing required fields")

        # Create a new Recipe object in the database
        # Only saving name and instructions for now (other fields not included)
        r = Recipe.objects.create(
            name=name,
            instructions=desc,
        )

        # Redirect user to the detail page of the newly created recipe
        # pk=r.pk passes the recipe's ID to the URL
        return redirect('recipe_detail', pk=r.pk)

