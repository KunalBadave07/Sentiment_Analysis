from django.urls import path
from Sentiment_app import views

urlpatterns = [
    path("", views.index, name="index"),
    path("Instagram", views.Instagram, name="Instagram"),
    path("Twitter", views.Twitter, name="Twitter"),
    path("WhatsApp", views.WhatsApp, name="WhatsApp"),
    path("Youtube", views.analyze_sentiment_view, name="Youtube"),
    path("analyze_sentiment/", views.analyze_sentiment_view, name="analyze_sentiment"),  # Add this line
    path("Reddit", views.Reddit, name='Reddit'),
]
