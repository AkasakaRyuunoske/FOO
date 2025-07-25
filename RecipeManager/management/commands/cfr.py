import csv
import os
from collections import defaultdict

from RecipeManager.models import Recipe
from TagManager.models import Tag
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand

from TagManager.models import RecipeTag


def build_tag_mapping():
    tags_by_type_and_name = defaultdict(dict)

    for tag in Tag.objects.select_related("tag_type").filter(tag_type__isnull=False):
        tag_type_name = tag.tag_type.name.strip().lower()
        tag_name = tag.name.strip().lower()
        tags_by_type_and_name[tag_type_name][tag_name] = tag

    return tags_by_type_and_name


class Command(BaseCommand):
    help = "Load recipes from a large CSV file into the database"

    def add_arguments(self, parser):
        parser.add_argument("--nrows", default=5, type=int, help="Number of rows to insert")

    def handle(self, *args, **options):
        nrows = options["nrows"]
        count = 0

        csv_path = os.path.join(settings.BASE_DIR, "TagManager", "mvp_tagging", "full_tagged_dataset_10%.csv")

        self.create_default_user()
        tags = build_tag_mapping()

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if count >= nrows:
                        self.stdout.write(f"Stopping at {count} inserted recipes...")
                        break

                    recipe = Recipe.objects.create(
                        name=row["TITLE"],
                        Instructions=row["DIRECTIONS"],
                    )
                    count += 1

                    RecipeTag.objects.create(recipe=recipe, tag=tags["difficulty"][row["difficulty"].strip().lower()])
                    RecipeTag.objects.create(recipe=recipe, tag=tags["preparation time"][row["PREPARATION_TIME"].split('(')[0].strip().lower()])
                    RecipeTag.objects.create(recipe=recipe, tag=tags["vegetarian"]["yes" if row["vegetarian"] else "no"])
                    RecipeTag.objects.create(recipe=recipe, tag=tags["vegan"]["yes" if row["vegan"] else "no"])
                    RecipeTag.objects.create(recipe=recipe, tag=tags["cooking method"][row["method"].strip().lower()])
                    RecipeTag.objects.create(recipe=recipe, tag=tags["cost"][row["price_tag"].strip().lower()])

                except Exception as exception:
                    self.stderr.write(f"Skipping row due to error: {exception}")
                    continue

        self.stdout.write(self.style.SUCCESS(f"Successfully inserted {count} recipes."))

    def create_default_user(self):
        User = get_user_model()

        try:
            User.objects.create_user(
                username="System",
                email="system@admin.com",
                password="dontstealmypasswordple34s"
            )
        except Exception as exception:
            self.stderr.write(f"Skipping user creation due to error: {exception}")
