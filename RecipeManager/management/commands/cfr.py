import csv
import os

from django.conf import settings
from django.core.management.base import BaseCommand
from RecipeManager.models import Recipe


class Command(BaseCommand):
    help = "Load recipes from a large CSV file into the database"

    def add_arguments(self, parser):
        parser.add_argument("nrows", type=int, help="Number of rows to insert")

    def handle(self, *args, **options):
        nrows = options["nrows"]
        count = 0

        csv_path = os.path.join(settings.BASE_DIR, "TagManager", "mvp_tagging", "full_tagged_dataset_10%.csv")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if count >= nrows:
                        self.stdout.write(f"Stopping at {count} inserted recipes...")
                        break

                    Recipe.objects.create(
                        name=row["TITLE"],
                        Instructions=row["DIRECTIONS"],
                        # TODO: Aggiungere altri campi appena verano create le entita neccessarie
                    )
                    count += 1

                    if count % 1000 == 0:
                        self.stdout.write(f"Inserted {count} recipes...")
                        break
                except Exception as e:
                    self.stderr.write(f"Skipping row due to error: {e}")
                    continue

        self.stdout.write(self.style.SUCCESS(f"Successfully inserted {count} recipes."))
