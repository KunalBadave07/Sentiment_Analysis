from django.shortcuts import render, HttpResponse
from googleapiclient.discovery import build
from .sentiment_analysis import (
    get_all_video_comments_with_ids,
    extract_video_id,
    evaluate_sentiment,
)
from django import forms
from django.http import JsonResponse


# Create your views here.
def index(request):
    return render(request, "index.html")


def Instagram(request):
    return render(request, "Instagram.html")


def Twitter(request):
    return render(request, "twitter.html")

def Reddit(request):
    return render(request, "Reddit.html")

def WhatsApp(request):
    return render(request, "WhatsApp.html")


class YouTubeURLForm(forms.Form):
    video_url = forms.URLField(label="Enter a YouTube video URL", max_length=100)


def analyze_sentiment_view(request):
    if request.method == "POST":
        form = YouTubeURLForm(request.POST)
        if form.is_valid():
            video_url = form.cleaned_data.get("video_url")
            video_id = extract_video_id(video_url)
            comments = get_all_video_comments_with_ids(video_id)
            comments_text = " ".join([comment["text"] for comment in comments])
            # Assuming `evaluate_sentiment` function expects `comments_text` as input
            result = evaluate_sentiment(comments_text)
            # return HttpResponse(
            #     f"Sentiment: {result['label']}, Score: {result['score']}"
            return JsonResponse({"label": result["label"], "score": result["score"]})
    else:
        form = YouTubeURLForm()
        return render(request, "Youtube.html", {"form": form})
    