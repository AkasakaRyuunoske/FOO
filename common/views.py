from django.shortcuts import render

def home(request):
    return render(request, "home.html", None)

def discover(request):
    return render(request, "discover.html", None)
