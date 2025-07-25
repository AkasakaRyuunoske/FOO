from django.contrib.auth.decorators import login_required
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from django.contrib.auth import login
from .forms import CustomUserCreationForm


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


@login_required
def user_profile(request):
    return render(request, "user_profile.html", {"user": request.user})