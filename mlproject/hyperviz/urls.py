from django.urls import path
from .views import index, regressor

urlpatterns = [
    path('classify/', index, name='index'),
    path('regressor/', regressor, name='regressor'),
]
