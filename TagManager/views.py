from collections import defaultdict

from RecipeManager.models import Recipe
from django.core.paginator import Paginator
from django.db.models import Count, Q
from django.shortcuts import render

from TagManager.models import Tag


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
