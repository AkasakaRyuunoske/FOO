"""
URL configuration for foo project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

from common import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('users/', include('django.contrib.auth.urls')),
    path('users/', include('UserManager.urls')),
    path('home/', views.home, name='home'),
    path('discover/', views.discover, name='discover'),
    path('', include("common.urls")),
    # path('recipe/', include('RecipeManager.urls')),
    path('tag/', include('TagManager.urls')),
    # path('ingredient/', include('IngredientManager.urls')),
    # path('activity/', include('UserActivityManager.urls')),
]
