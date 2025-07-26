from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.core.paginator import Paginator
from .forms import CustomUserCreationForm
from RecipeManager.models import Recipe

def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = CustomUserCreationForm()
    return render(request, 'registration/register.html', {'form': form})

# @login_required
def user_profile(request):
    return render(request, "user_profile.html", {"user": request.user})

# @login_required
# def user_profile(request):
#     user = request.user
#     # how many recipes they've published
#     published_count = Recipe.objects.filter(author=user).count()
#     # default tab: saved recipes
#     saved_list = user.saved_recipes.all().order_by('-pk')
#     paginator = Paginator(saved_list, 12)
#     page_obj = paginator.get_page(request.GET.get('page'))
#
#     return render(request, 'user_profile.html', {
#         'published_count': published_count,
#         'saved_page_obj': page_obj,
#     } )

@login_required
def get_saved_recipe(request):
    # HTMX endpoint for the “Saved Recipes” tab
    user = request.user
    saved_list = user.saved_recipes.all().order_by('-pk')
    paginator = Paginator(saved_list, 12)
    page_obj = paginator.get_page(request.GET.get('page'))
    return render(request, 'components/recipes_list.html', {
        'page_obj': page_obj
    })

@login_required
def get_activity(request):
    user = request.user
    activity_qs = Recipe.objects.filter(author=user).order_by('-created_at')
    paginator = Paginator(activity_qs, 12)
    page_obj = paginator.get_page(request.GET.get('page'))
    return render(request, 'activity_list.html', {
        'page_obj': page_obj
    })
