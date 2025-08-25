from collections import defaultdict

from RecipeManager.models import Recipe
from django.core.paginator import Paginator
from django.db.models import Count, Q
from django.shortcuts import render

from TagManager.models import Tag

from FOO.common.views import ICON_MAPPING, normalize_tag_value, boolean_from_name, EXCLUDED_TYPES


def build_tag_mapping():
    tags_by_type_and_name = defaultdict(dict)

    for tag in Tag.objects.select_related("tag_type").filter(tag_type__isnull=False):
        tag_type_name = tag.tag_type.name.strip().lower()
        tag_name = tag.name.strip().lower()
        tags_by_type_and_name[tag_type_name][tag_name] = tag

    return tags_by_type_and_name


def get_tags(request):
    tags_by_type_and_name = dict(build_tag_mapping())
    return render(request, "components/tags_list.html", {"tags_by_type_and_name": tags_by_type_and_name})


def get_recipes_by_tags(request):
    tag_string = request.GET.get("tags", "")
    tags = [t.strip() for t in tag_string.split(",") if t.strip()]

    if tags:
        tag_ids = Tag.objects.filter(name__in=tags).values_list("id", flat=True)

        recipes = Recipe.objects.annotate(
            matched_tags=Count(
                "recipetag", filter=Q(recipetag__tag_id__in=tag_ids), distinct=True
            )
        ).filter(matched_tags=len(tags))
    else:
        recipes = Recipe.objects.all()

    # Paginazione
    page_number = request.GET.get("page", 1)
    paginator = Paginator(recipes, 12)
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

    return render(request, "components/recipe_cards.html", {
        "page_obj": page_obj,
        "tag_pairs_by_recipe": tag_pairs_by_recipe,
        "meta_pairs_by_recipe": meta_pairs_by_recipe,
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
