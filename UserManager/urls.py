from django.contrib.auth.views import LogoutView
from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register, name='register'),
    path("logout/", LogoutView.as_view(next_page="register"), name="logout"),
    path("profile/", views.user_profile, name="user_profile"),
]