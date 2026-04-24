from django.shortcuts import render
from .sentiment_analysis import (
    analyze_comments,
    analyze_text,
    get_all_video_comments_with_ids,
    extract_video_id,
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
    video_url = forms.URLField(label="Enter a YouTube video URL", max_length=250)


def analyze_sentiment_view(request):
    if request.method == "POST":
        form = YouTubeURLForm(request.POST)
        if form.is_valid():
            video_url = form.cleaned_data.get("video_url")
            video_id = extract_video_id(video_url)
            if not video_id:
                return JsonResponse(
                    {"error": "Please paste a valid YouTube video, shorts, embed, or youtu.be link."},
                    status=400,
                )

            try:
                comments = get_all_video_comments_with_ids(video_id)
                result = analyze_comments(comments)
            except ValueError as exc:
                return JsonResponse({"error": str(exc)}, status=400)

            return JsonResponse(result)

        return JsonResponse({"error": "Please paste a valid YouTube URL."}, status=400)

    form = YouTubeURLForm()
    return render(request, "Youtube.html", {"form": form})


def analyze_text_sentiment_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST text to analyze sentiment."}, status=405)

    text = request.POST.get("text", "")
    try:
        result = analyze_text(text)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    return JsonResponse(result)
