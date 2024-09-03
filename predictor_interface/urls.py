from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('train/', views.train_model, name='train'),
    path('predict/', views.predict_model, name='predict')
]
