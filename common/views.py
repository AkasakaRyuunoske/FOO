from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request, "home.html", None)

def discover(request):
    return render(request, "discover.html", None)

def recipe_details(request):
    return render(request, "recipe_details.html", None)
