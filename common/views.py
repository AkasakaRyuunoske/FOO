import json
import os
import pickle
import random
import re

from django.core.paginator import Paginator
from django.db.models import Prefetch
from django.http import HttpResponseBadRequest
from django.shortcuts import render, get_object_or_404
from django.views import View
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from RecipeManager.models import Recipe
from TagManager.models import Tag, RecipeTag

ICON_MAPPING = {
    "Cooking Method": {
        "Fried": "fry.png",
        "Baked": "bake.png",
        "Grilled": "grill.png",
        "Boiled": "boil.png",
        "Blended": "blender.png",
        "Microwaved": "blender.png",
        "Pressure Cooked": "pressure_cooker.png",
        "Steamed": "steam.png",
        "Raw": "uncooked.png",
    },
    "Difficulty": {
        "Easy": "difficulty.png",
        "Medium": "difficulty.png",
        "Hard": "difficulty.png"
    },
    "Preparation Time": {
        "Very Fast": "time.png",
        "Fast": "time.png",
        "Medium": "time.png",
        "Slow": "time.png",
        "Very Slow": "time.png",
    },
    "Cost": {
        "Very Cheap": "cost.png",
        "Cheap": "cost.png",
        "Medium": "cost.png",
        "Expensive": "cost.png",
        "Rich": "cost.png",
    },
    "Vegan": {
        True: "vegan.png",
        False: "meat.png"
    },
    "Vegetarian": {
        True: "vegetarian.png",
        False: "meat.png"
    },
    "Lactose Free": {
        True: "lactose-free.png",
        False: "cheese.png"
    },
    "Gluten Free": {
        True: "gluten-free.png",
        False: "bread.png"
    }
}
EXCLUDED_TYPES = {"Cost", "Difficulty", "Preparation Time"}


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

    # Costruzione mappe: recipe_id -> dict di coppie {tag_type: {icon,label}}
    tag_pairs_by_recipe = {}
    meta_pairs_by_recipe = {}

    for recipe in page_obj.object_list:
        display_pairs = {}
        meta_pairs = {}

        for rt in recipe.recipe_tags.all():
            tag = rt.tag
            tag_type_name = tag.tag_type.name
            raw_tag_value = tag.name

            tag_value = boolean_from_name(raw_tag_value)
            if isinstance(tag_value, str):
                tag_value = normalize_tag_value(tag_value)

            icon = ICON_MAPPING.get(tag_type_name, {}).get(tag_value, "hat.png")
            pair = {"icon": icon, "label": raw_tag_value}

            if tag_type_name in EXCLUDED_TYPES:
                meta_pairs[tag_type_name] = pair
            else:
                display_pairs[tag_type_name] = pair

        tag_pairs_by_recipe[recipe.id] = display_pairs
        meta_pairs_by_recipe[recipe.id] = meta_pairs

    return render(request, "home.html", {
        "page_obj": page_obj,
        "tag_pairs_by_recipe": tag_pairs_by_recipe,
        "meta_pairs_by_recipe": meta_pairs_by_recipe,
    })


def recipe_list(request):
    recipes_qs = list(Recipe.objects.all().prefetch_related(
        Prefetch(
            "recipe_tags",
            queryset=RecipeTag.objects.select_related("tag", "tag__tag_type")
        ),
        "ratings"
    ))

    paginator = Paginator(recipes_qs, 10)
    page_number = request.GET.get("page", 1)
    page_obj = paginator.get_page(page_number)

    tag_pairs_by_recipe = {}
    meta_pairs_by_recipe = {}
    rating_by_recipe = {}

    for recipe in page_obj.object_list:
        display_pairs = {}
        meta_pairs = {}

        for rt in recipe.recipe_tags.all():
            tag = rt.tag
            tag_type_name = tag.tag_type.name
            raw_tag_value = tag.name

            tag_value = boolean_from_name(raw_tag_value)
            if isinstance(tag_value, str):
                tag_value = normalize_tag_value(tag_value)

            icon = ICON_MAPPING.get(tag_type_name, {}).get(tag_value, "hat.png")
            pair = {"icon": icon, "label": raw_tag_value}

            if tag_type_name in EXCLUDED_TYPES:
                meta_pairs[tag_type_name] = pair
            else:
                display_pairs[tag_type_name] = pair

        tag_pairs_by_recipe[recipe.id] = display_pairs
        meta_pairs_by_recipe[recipe.id] = meta_pairs

        rating = recipe.ratings.first()
        rating_by_recipe[recipe.id] = rating.stars if rating else 0

        print(f"rating_by_recipe ==> {rating_by_recipe}")
        print(f"tag_pairs_by_recipe ==> {tag_pairs_by_recipe}")

    return render(request, "discover.html", {
        "page_obj": page_obj,
        "tag_pairs_by_recipe": tag_pairs_by_recipe,
        "meta_pairs_by_recipe": meta_pairs_by_recipe,
        "rating_by_recipe": rating_by_recipe,
    })

def discover(request):
    # Prefetch tag e tag_type in un’unica query aggiuntiva
    recipes_qs = Recipe.objects.prefetch_related(
        Prefetch(
            "recipe_tags",
            queryset=RecipeTag.objects.select_related("tag", "tag__tag_type")
        ),
        "ratings"
    )

    # Paginazione
    page_number = request.GET.get("page", 1)
    paginator = Paginator(recipes_qs, 12)
    page_obj = paginator.get_page(page_number)

    # Costruzione mappe: recipe_id -> dict di coppie {tag_type: {icon,label}}
    tag_pairs_by_recipe = {}
    meta_pairs_by_recipe = {}

    rating_by_recipe = {}

    for recipe in page_obj.object_list:
        display_pairs = {}
        meta_pairs = {}

        for rt in recipe.recipe_tags.all():
            tag = rt.tag
            tag_type_name = tag.tag_type.name
            raw_tag_value = tag.name

            tag_value = boolean_from_name(raw_tag_value)
            if isinstance(tag_value, str):
                tag_value = normalize_tag_value(tag_value)

            icon = ICON_MAPPING.get(tag_type_name, {}).get(tag_value, "hat.png")
            pair = {"icon": icon, "label": raw_tag_value}

            if tag_type_name in EXCLUDED_TYPES:
                meta_pairs[tag_type_name] = pair
            else:
                display_pairs[tag_type_name] = pair

        tag_pairs_by_recipe[recipe.id] = display_pairs
        meta_pairs_by_recipe[recipe.id] = meta_pairs

        rating = recipe.ratings.first()
        rating_by_recipe[recipe.id] = rating.stars if rating else 0

        print(f"rating_by_recipe ==> {rating_by_recipe}")
        print(f"tag_pairs_by_recipe ==> {tag_pairs_by_recipe}")
    return render(request, "discover.html", {
        "page_obj": page_obj,
        "tag_pairs_by_recipe": tag_pairs_by_recipe,
        "meta_pairs_by_recipe": meta_pairs_by_recipe,
        "rating_by_recipe": rating_by_recipe,
    })


def random_recipe(request):
    all_ids = Recipe.objects.values_list('id', flat=True)

    rand_id = random.choice(list(all_ids))
    recipe = Recipe.objects.get(pk=rand_id)

    recipes = [recipe]

    paginator = Paginator(recipes, per_page=1)
    page_obj = paginator.page(1)

    recipe_tags = RecipeTag.objects.filter(recipe=recipe).select_related('tag')

    EXCLUDED_TYPES = {"Cost", "Difficulty", "Preparation Time"}

    display_tag_icon_pairs = {}
    meta_tag_icon_pairs = {}

    for rt in recipe_tags:
        tag = rt.tag
        tag_type_name = tag.tag_type.name
        raw_tag_value = tag.name

        tag_value = boolean_from_name(raw_tag_value)
        if isinstance(tag_value, str):
            tag_value = normalize_tag_value(tag_value)

        icon = ICON_MAPPING.get(tag_type_name, {}).get(tag_value, "hat.png")

        pair = {
            "icon": icon,
            "label": raw_tag_value
        }

        if tag_type_name in EXCLUDED_TYPES:
            meta_tag_icon_pairs[tag_type_name] = pair
        else:
            display_tag_icon_pairs[tag_type_name] = pair

    return render(request, "components/recipe_cards.html", {
        'page_obj': page_obj,
        "tag_icon_pairs": display_tag_icon_pairs,
        "meta_tag_icon_pairs": meta_tag_icon_pairs
    })


def normalize_tag_value(name):
    return name.split("(")[0].strip()


def boolean_from_name(name):
    if name.strip().lower() == "yes":
        return True
    if name.strip().lower() == "no":
        return False
    return name

def parse_instruction_steps(instructions_raw):
    if not instructions_raw:
        return []

    try:
        parsed = json.loads(instructions_raw)
        if isinstance(parsed, list):
            return [s.strip() for s in parsed if isinstance(s, str) and s.strip()]
    except json.JSONDecodeError:
        pass

    return re.split(r'(?<=[.?!])\s+(?=[A-Z])', instructions_raw.strip())

def load_recipe_modal(request, recipe_id):
    recipe = get_object_or_404(
        Recipe.objects.prefetch_related(
            "ingredient_links__ingredient", "recipe_tags__tag__tag_type"
        ),
        id=recipe_id
    )

    # Costruzione tag icon pairs
    display_pairs = {}
    meta_pairs = {}

    for rt in recipe.recipe_tags.all():
        tag = rt.tag
        tag_type_name = tag.tag_type.name
        raw_tag_value = tag.name

        tag_value = boolean_from_name(raw_tag_value)
        if isinstance(tag_value, str):
            tag_value = normalize_tag_value(tag_value)

        icon = ICON_MAPPING.get(tag_type_name, {}).get(tag_value, "hat.png")
        pair = {"icon": icon, "label": raw_tag_value}

        if tag_type_name in EXCLUDED_TYPES:
            meta_pairs[tag_type_name] = pair
        else:
            display_pairs[tag_type_name] = pair

    rating = recipe.ratings.first()
    rating_value = rating.stars if rating else 0

    return render(request, "components/recipe_modal.html", {
        "recipe": recipe,
        "steps": parse_instruction_steps(recipe.Instructions),
        "rating_value": rating_value,
        "tag_icon_pairs": display_pairs,
        "meta_tag_icon_pairs": meta_pairs
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
        if not all([name, cooking_time_unit]):
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

        difficulty_predictions = predict_and_print(recipe, difficulty_model, difficulty_tokenizer,
                                                   inverse_difficulty_map, "Difficulty")
        time_predictions = predict_and_print(recipe, time_model, time_tokenizer, inverse_time_map, "Difficulty")
        gluten_free_predictions = predict_and_print(recipe, gluten_free_model, gluten_free_tokenizer,
                                                    inverse_gluten_free_map, "Gluten Free")
        lactose_free_predictions = predict_and_print(recipe, lactose_free_model, lactose_free_tokenizer,
                                                     inverse_lactose_free_map, "Lactose Free")
        cooking_method_predictions = predict_and_print(recipe, cooking_method_model, cooking_method_tokenizer,
                                                       inverse_cooking_method_map, "Method")
        price_predictions = predict_and_print(recipe, price_model, price_tokenizer, inverse_price_map, "Price")
        vegan_predictions = predict_and_print(recipe, vegan_model, vegan_tokenizer, inverse_vegan_map, "Vegan")
        vegetarian_predictions = predict_and_print(recipe, vegetarian_model, vegetarian_tokenizer,
                                                   inverse_vegetarian_map, "Vegetarian")

        RecipeTag.objects.create(recipe=recipe_obj, tag=Tag.objects.get(name=difficulty_predictions["predicted"],
                                                                        tag_type__name="Difficulty"))
        RecipeTag.objects.create(recipe=recipe_obj,
                                 tag=Tag.objects.get(name=time_predictions["predicted"].split('(')[0].strip(),
                                                     tag_type__name="Preparation Time"))
        RecipeTag.objects.create(recipe=recipe_obj,
                                 tag=Tag.objects.get(name="Yes" if gluten_free_predictions["predicted"] else "No",
                                                     tag_type__name="Gluten Free"))
        RecipeTag.objects.create(recipe=recipe_obj,
                                 tag=Tag.objects.get(name="Yes" if lactose_free_predictions["predicted"] else "No",
                                                     tag_type__name="Lactose Free"))
        RecipeTag.objects.create(recipe=recipe_obj, tag=Tag.objects.get(name=cooking_method_predictions["predicted"],
                                                                        tag_type__name="Cooking Method"))
        RecipeTag.objects.create(recipe=recipe_obj,
                                 tag=Tag.objects.get(name=price_predictions["predicted"], tag_type__name="Cost"))
        RecipeTag.objects.create(recipe=recipe_obj,
                                 tag=Tag.objects.get(name="Yes" if vegan_predictions["predicted"] else "No",
                                                     tag_type__name="Vegan"))
        RecipeTag.objects.create(recipe=recipe_obj,
                                 tag=Tag.objects.get(name="Yes" if vegetarian_predictions["predicted"] else "No",
                                                     tag_type__name="Vegetarian"))

        predictions = {"difficulty_predictions": difficulty_predictions,
                       "time_predictions": time_predictions,
                       "gluten_free_predictions": gluten_free_predictions,
                       "lactose_free_predictions": lactose_free_predictions,
                       "cooking_method_predictions": cooking_method_predictions,
                       "price_predictions": price_predictions,
                       "vegan_predictions": vegan_predictions,
                       "vegetarian_predictions": vegetarian_predictions,
                       }

        map_icons(predictions)

        # Redirect user to the detail page of the newly created recipe
        return render(request, "components/recipe_created_success.html",
                      {"recipe": recipe_obj, "predictions": predictions})


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

    return {"predicted": pred_labels[0], "probabilities": probs_str, "prediction_label": prediction_label}


def load_model_components(model_dir, model_name_prefix):
    model = load_model(f"{model_dir}/{model_name_prefix}.h5")

    with open(f"{model_dir}/{model_name_prefix}_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open(f"{model_dir}/{model_name_prefix}_label_mapping.pkl", "rb") as f:
        label_map = pickle.load(f)

    # Reverse dictionaries to decode predictions
    inv_label_map = {v: k for k, v in label_map.items()}
    return model, tokenizer, inv_label_map


def map_icons(predictions):
    for pred in predictions.values():
        label = pred["prediction_label"]
        value = pred["predicted"]
        pred["icon"] = ICON_MAPPING.get(label, {}).get(value, "hat.png")
