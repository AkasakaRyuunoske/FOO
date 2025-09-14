from django.core.paginator import Paginator
from django.db.models import Count, Q
from django.shortcuts import render

from IngredientManager.models import Ingredient
from RecipeManager.models import Recipe
from TagManager.models import Tag
from common.views import EXCLUDED_TYPES, ICON_MAPPING, normalize_tag_value, boolean_from_name


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

def get_recipes_by_ingredients(request):
    ingredient_string = request.GET.get("ingredients", "")
    tag_string = request.GET.get("tags", "")

    ingredient_names = [i.strip().lower() for i in ingredient_string.split(",") if i.strip()]
    tag_names = [t.strip() for t in tag_string.split(",") if t.strip()]

    recipes = Recipe.objects.all()

    # Filtro per ingredienti
    if ingredient_names:
        ingredients = Ingredient.objects.filter(name__in=ingredient_names)

        if ingredients.count() != len(ingredient_names):
            return render(request, "components/recipe_cards.html", {
                "page_obj": Paginator(Recipe.objects.none(), 12).get_page(1),
                "tag_pairs_by_recipe": {},
                "meta_pairs_by_recipe": {},
            })

        recipes = recipes.filter(
            ingredient_links__ingredient__in=ingredients
        ).annotate(
            matched_ingredients=Count(
                "ingredient_links",
                filter=Q(ingredient_links__ingredient__in=ingredients),
                distinct=True,
            )
        ).filter(matched_ingredients=len(ingredient_names))

    # Filtro per tag
    if tag_names:
        tag_ids = Tag.objects.filter(name__in=tag_names).values_list("id", flat=True)

        recipes = recipes.annotate(
            matched_tags=Count(
                "recipe_tags",
                filter=Q(recipe_tags__tag_id__in=tag_ids),
                distinct=True,
            )
        ).filter(matched_tags=len(tag_ids))

    # Paginazione
    page_number = request.GET.get("page", 1)
    paginator = Paginator(recipes, 12)
    page_obj = paginator.get_page(page_number)

    # Tag associati
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

    return render(request, "components/recipe_cards.html", {
        "page_obj": page_obj,
        "tag_pairs_by_recipe": tag_pairs_by_recipe,
        "meta_pairs_by_recipe": meta_pairs_by_recipe,
    })