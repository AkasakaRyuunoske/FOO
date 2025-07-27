from django.contrib.auth.views import LogoutView
from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register, name='register'),
    path('logout/', LogoutView.as_view(next_page='register'), name='logout'),
    path('profile/', views.user_profile, name='user_profile'),
    path('profile/saved/', views.get_saved_recipe, name='saved_recipes'),
    path('profile/activity/', views.get_activity,   name='profile_activity'),
]
