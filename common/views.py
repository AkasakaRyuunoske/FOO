import json
import random
import os
import pickle

from RecipeManager.models import Recipe
from TagManager.models import Tag
from django.core.paginator import Paginator
from django.http import HttpResponseBadRequest, JsonResponse
from django.shortcuts import render, redirect
from django.views import View
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


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
        tags = Tag.objects.all()
        return render(request, self.template_name, context={"tags": tags})

    def post_recipe_form(self, request):
        """
        Processes the submitted form data and creates a new recipe
        This is called when the user clicks submit on the form
        """
        # Extract data from the submitted form
        # request.POST.get() safely gets form field values
        name = request.POST.get('name')
        description = request.POST.get('description')   # TODO ricordare di aggiungere eventualmente
        instructions = request.POST.get('instructions')
        cooking_time = request.POST.get('cook_time_value')
        cooking_time_unit = request.POST.get('cook_time_unit')
        ingredients_data = json.loads(request.POST.get("ingredients_json", "[]"))

        print(f"ingredients data ==> {ingredients_data}")
        # Check if all required fields have values
        # all() returns True only if all items in the list are truthy (not empty)
        if not all([name, cooking_time, cooking_time_unit]):
            # Return error response if any field is missing
            return HttpResponseBadRequest("Missing required fields")

        # Create a new Recipe object in the database
        # Only saving name and instructions for now (other fields not included)
        recipe = Recipe.objects.create(
            name=name,
            Instructions=instructions,
        )

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_DIR = os.path.join(BASE_DIR, 'TagManager/mvp_tagging/classifiers')

        # Carica tokenizer pickle
        with open(os.path.join(MODEL_DIR, 'difficulty', 'difficulty_classifier_tokenizer.pkl'), 'rb') as f:
            difficulty_tokenizer = pickle.load(f)

        with open(os.path.join(MODEL_DIR, 'prep_time', 'prep_time_classifier_tokenizer.pkl'), 'rb') as f:
            time_tokenizer = pickle.load(f)

        # Carica modelli
        difficulty_model = load_model(os.path.join(MODEL_DIR, 'difficulty', 'difficulty_classifier.h5'))
        time_model = load_model(os.path.join(MODEL_DIR, 'prep_time', 'prep_time_classifier.h5'))

        ingredients = ["potatoes", "fish", "peaches", "banana"]

        # Redirect user to the detail page of the newly created recipe
        return render(request, "components/recipe_created_success.html", {"recipe": recipe})

def preprocess_text(text, tokenizer):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=500)
    return padded

def get_difficulty_tag(predicted_label):
    return Tag.objects.filter(type='Difficulty', name__iexact=predicted_label).first()