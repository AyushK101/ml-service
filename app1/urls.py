from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('health/',views.health,name="health" ),
     path('calculate-solar/', views.calculate_solar, name='calculate_solar'),
    path('predict_view/', views.predict_view, name='predict_view'),
    path('gen/', views.gen, name='gen'),
]