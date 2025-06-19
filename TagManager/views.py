import json
from time import sleep

from django.http import HttpResponse
from django.shortcuts import render


def get_tags(request):
    sleep(3)

    result = ["Fast", "Slow", "Average", "Overnight",
              "Cheap", "Average", "Expensive", "Rich",
              "Vegan", "Vegetarian",
              "Boiled", "Grilled", "Backed", "Boiled",
              "Easy", "Medium", "Hard", "Chef"]

    return render(request, "components/item.html", {"tags": result})
