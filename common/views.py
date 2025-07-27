import json
import os
import pickle
import random

from django.core.paginator import Paginator
from django.http import HttpResponseBadRequest
from django.shortcuts import render
from django.views import View
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from RecipeManager.models import Recipe
from TagManager.models import Tag, RecipeTag


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
        description = request.POST.get('description')  # TODO ricordare di aggiungere eventualmente
        instructions = request.POST.get('instructions')
        cooking_time = request.POST.get('cook_time_value')
        cooking_time_unit = request.POST.get('cook_time_unit')
        ingredients_data = json.loads(request.POST.get("ingredients_json"))

        # Check if all required fields have values
        # all() returns True only if all items in the list are truthy (not empty)
        if not all([name, cooking_time, cooking_time_unit]):
            # Return error response if any field is missing
            return HttpResponseBadRequest("Missing required fields")

        # Create a new Recipe object in the database
        # Only saving name and instructions for now (other fields not included)
        recipe_obj = Recipe.objects.create(
            name=name,
            Instructions=instructions,
        )

        ingredients = []
        for ingredient in ingredients_data:
            ingredients.append(f"{ingredient['quantity']}{ingredient['unit']} {ingredient['name']}")

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_DIR = os.path.join(BASE_DIR, 'TagManager/mvp_tagging/classifiers')

        difficulty_model, difficulty_tokenizer, inverse_difficulty_map = load_model_components(
            os.path.join(MODEL_DIR, "difficulty"), "difficulty_classifier")

        time_model, time_tokenizer, inverse_time_map = load_model_components(
            os.path.join(MODEL_DIR, "prep_time"), "prep_time_classifier")

        gluten_free_model, gluten_free_tokenizer, inverse_gluten_free_map = load_model_components(
            os.path.join(MODEL_DIR, "gluten_free"), "gluten_free_classifier")

        lactose_free_model, lactose_free_tokenizer, inverse_lactose_free_map = load_model_components(
            os.path.join(MODEL_DIR, "lactose_free"), "lactose_free_classifier")

        cooking_method_model, cooking_method_tokenizer, inverse_cooking_method_map = load_model_components(
            os.path.join(MODEL_DIR, "method"), "method_classifier")

        price_model, price_tokenizer, inverse_price_map = load_model_components(
            os.path.join(MODEL_DIR, "price"), "price_classifier")

        vegan_model, vegan_tokenizer, inverse_vegan_map = load_model_components(
            os.path.join(MODEL_DIR, "vegan"), "vegan_classifier")

        vegetarian_model, vegetarian_tokenizer, inverse_vegetarian_map = load_model_components(
            os.path.join(MODEL_DIR, "vegetarian"), "vegetarian_classifier")

        recipe = [{"ingredients": ingredients, "instructions": instructions}]

        difficulty_predictions = predict_and_print(recipe, difficulty_model, difficulty_tokenizer, inverse_difficulty_map, "Difficulty")
        time_predictions = predict_and_print(recipe, time_model, time_tokenizer, inverse_time_map, "Difficulty")
        gluten_free_predictions = predict_and_print(recipe, gluten_free_model, gluten_free_tokenizer, inverse_gluten_free_map, "Gluten Free")
        lactose_free_predictions = predict_and_print(recipe, lactose_free_model, lactose_free_tokenizer, inverse_lactose_free_map, "Lactose Free")
        cooking_method_predictions = predict_and_print(recipe, cooking_method_model, cooking_method_tokenizer, inverse_cooking_method_map, "Method")
        price_predictions = predict_and_print(recipe, price_model, price_tokenizer, inverse_price_map, "Price")
        vegan_predictions = predict_and_print(recipe, vegan_model, vegan_tokenizer, inverse_vegan_map, "Vegan")
        vegetarian_predictions = predict_and_print(recipe, vegetarian_model, vegetarian_tokenizer, inverse_vegetarian_map, "Vegetarian")

        RecipeTag.objects.create(recipe=recipe_obj, tag=Tag.objects.get(name=difficulty_predictions["predicted"], tag_type__name="Difficulty"))
        RecipeTag.objects.create(recipe=recipe_obj, tag=Tag.objects.get(name=time_predictions["predicted"].split('(')[0].strip(), tag_type__name="Preparation Time"))
        RecipeTag.objects.create(recipe=recipe_obj, tag=Tag.objects.get(name="Yes" if gluten_free_predictions["predicted"] else "No", tag_type__name="Gluten Free"))
        RecipeTag.objects.create(recipe=recipe_obj, tag=Tag.objects.get(name="Yes" if lactose_free_predictions["predicted"] else "No", tag_type__name="Lactose Free"))
        RecipeTag.objects.create(recipe=recipe_obj, tag=Tag.objects.get(name=cooking_method_predictions["predicted"], tag_type__name="Cooking Method"))
        RecipeTag.objects.create(recipe=recipe_obj, tag=Tag.objects.get(name=price_predictions["predicted"], tag_type__name="Cost"))
        RecipeTag.objects.create(recipe=recipe_obj, tag=Tag.objects.get(name="Yes" if vegan_predictions["predicted"] else "No", tag_type__name="Vegan"))
        RecipeTag.objects.create(recipe=recipe_obj, tag=Tag.objects.get(name="Yes" if vegetarian_predictions["predicted"] else "No", tag_type__name="Vegetarian"))

        # Redirect user to the detail page of the newly created recipe
        return render(request, "components/recipe_created_success.html", {"recipe": recipe})


def get_difficulty_tag(predicted_label):
    return Tag.objects.filter(type='Difficulty', name__iexact=predicted_label).first()


def predict_and_print(recipes, model, tokenizer, inv_label_map, prediction_label):
    # Prepare input texts for models (ingredients + instructions)
    texts = [" ".join(r["ingredients"]) + " " + r["instructions"] for r in recipes]

    # Tokenize and padding
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=500, padding='post', truncating='post')

    # Predicting with probabilities
    preds = model.predict(padded)
    pred_classes = preds.argmax(axis=1)
    pred_labels = [inv_label_map[c] for c in pred_classes]

    # Printing results
    print(f"  Predicted {prediction_label}: {pred_labels[0]}")
    probs_str = ", ".join([f"{inv_label_map[j]}: {preds[0][j] * 100:.2f}%" for j in range(len(preds[0]))])
    print(f"  {prediction_label} probabilities: {probs_str}\n")

    return {"predicted": pred_labels[0], "probabilities": probs_str}


def load_model_components(model_dir, model_name_prefix):
    model = load_model(f"{model_dir}/{model_name_prefix}.h5")

    with open(f"{model_dir}/{model_name_prefix}_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open(f"{model_dir}/{model_name_prefix}_label_mapping.pkl", "rb") as f:
        label_map = pickle.load(f)

    # Reverse dictionaries to decode predictions
    inv_label_map = {v: k for k, v in label_map.items()}
    return model, tokenizer, inv_label_map
