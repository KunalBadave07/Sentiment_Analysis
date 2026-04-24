import os
import re
from collections import Counter

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

youtubeAPIKey = os.getenv("YOUTUBE_API_KEY")


def get_all_video_comments_with_ids(video_id, max_comments=300):
    if not video_id:
        raise ValueError("Please enter a valid YouTube video link.")

    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
    except ImportError as exc:
        raise ValueError("Install project dependencies before fetching YouTube comments.") from exc
    if not youtubeAPIKey:
        raise ValueError("Set the YOUTUBE_API_KEY environment variable before fetching YouTube comments.")

    comments_with_ids = []
    next_page_token = None
    youtube = build("youtube", "v3", developerKey=youtubeAPIKey)

    while True:
        try:
            response = (
                youtube.commentThreads()
                .list(
                    part="snippet",
                    videoId=video_id,
                    textFormat="plainText",
                    pageToken=next_page_token,
                    maxResults=min(100, max_comments - len(comments_with_ids)),
                    order="relevance",
                )
                .execute()
            )
        except HttpError as exc:
            raise ValueError("Unable to fetch comments for this video. Check the link, API key, quota, or comment availability.") from exc

        for item in response.get("items", []):
            comment_id = item["snippet"]["topLevelComment"]["id"]
            comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments_with_ids.append({"id": comment_id, "text": comment_text})

        next_page_token = response.get("nextPageToken")
        if not next_page_token or len(comments_with_ids) >= max_comments:
            break

    return comments_with_ids


def extract_video_id(url):
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/|youtube\.com/embed/)([A-Za-z0-9_-]{11})",
        r"[?&]v=([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _get_vader_analyzer():
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


def _label_from_compound(compound):
    if compound >= 0.05:
        return "POSITIVE"
    if compound <= -0.05:
        return "NEGATIVE"
    return "NEUTRAL"


def _clean_comment(text):
    return re.sub(r"\s+", " ", text).strip()


def analyze_comments(comments):
    if not comments:
        return {
            "label": "NEUTRAL",
            "score": 0,
            "confidence": 0,
            "total_comments": 0,
            "percentages": {"positive": 0, "neutral": 0, "negative": 0},
            "counts": {"positive": 0, "neutral": 0, "negative": 0},
            "insights": ["No public comments were available for analysis."],
            "top_comments": [],
        }

    analyzer = _get_vader_analyzer()
    scored_comments = []

    for comment in comments:
        text = _clean_comment(comment.get("text", ""))
        if not text:
            continue

        scores = analyzer.polarity_scores(text)
        label = _label_from_compound(scores["compound"])
        scored_comments.append(
            {
                "id": comment.get("id", ""),
                "text": text[:240],
                "label": label,
                "compound": round(scores["compound"], 4),
                "positive": round(scores["pos"], 4),
                "neutral": round(scores["neu"], 4),
                "negative": round(scores["neg"], 4),
            }
        )

    if not scored_comments:
        return analyze_comments([])

    total = len(scored_comments)
    counts = Counter(item["label"].lower() for item in scored_comments)
    percentages = {
        "positive": round((counts.get("positive", 0) / total) * 100, 1),
        "neutral": round((counts.get("neutral", 0) / total) * 100, 1),
        "negative": round((counts.get("negative", 0) / total) * 100, 1),
    }

    average_score = round(sum(item["compound"] for item in scored_comments) / total, 4)
    strongest_label = max(percentages, key=percentages.get).upper()
    label = _label_from_compound(average_score)
    if label == "NEUTRAL" and percentages[strongest_label.lower()] >= 45:
        label = strongest_label

    confidence = round(percentages[label.lower()], 1)
    insights = _build_insights(label, average_score, percentages, total)
    top_comments = sorted(scored_comments, key=lambda item: abs(item["compound"]), reverse=True)[:5]

    return {
        "label": label,
        "score": average_score,
        "confidence": confidence,
        "total_comments": total,
        "percentages": percentages,
        "counts": {
            "positive": counts.get("positive", 0),
            "neutral": counts.get("neutral", 0),
            "negative": counts.get("negative", 0),
        },
        "insights": insights,
        "top_comments": top_comments,
    }


def analyze_text(text):
    text = _clean_comment(text or "")
    if not text:
        raise ValueError("Please enter text to analyze.")

    pieces = [
        piece.strip()
        for piece in re.split(r"(?:[\r\n]+|(?<=[.!?])\s+)", text)
        if piece.strip()
    ]
    if not pieces:
        pieces = [text]

    comments = [{"id": str(index), "text": piece} for index, piece in enumerate(pieces[:80], start=1)]
    result = analyze_comments(comments)
    result["source_type"] = "text"
    return result


def _build_insights(label, average_score, percentages, total):
    dominant_percent = percentages[label.lower()]
    mixed_gap = abs(percentages["positive"] - percentages["negative"])
    insights = [
        f"Analyzed {total} public comments and classified each comment separately.",
        f"Overall sentiment is {label.title()} with an average compound score of {average_score}.",
    ]

    if dominant_percent >= 60:
        insights.append(f"{label.title()} comments clearly dominate at {dominant_percent}%.")
    elif mixed_gap <= 12 and percentages["neutral"] < 45:
        insights.append("Audience reaction is mixed, with positive and negative comments close to each other.")
    elif percentages["neutral"] >= 45:
        insights.append("A large neutral share suggests many comments are factual, short, or low-emotion.")
    else:
        insights.append("Sentiment is present but not extremely one-sided, so review the sample comments for context.")

    return insights


def poliarty_scores_roberta(example):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    encoded_text = tokenizer(example, return_tensors="pt")
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores_dict = {
        "roberta_neg": scores[0],
        "roberts_neu": scores[1],
        "roberta_pos": scores[2],
    }
    return scores_dict


def evaluate_sentiment(comments_text):
    analyzer = _get_vader_analyzer()
    scores = analyzer.polarity_scores(comments_text)
    return {"label": _label_from_compound(scores["compound"]), "score": scores["compound"]}


# def evaluate_sentiment(vader_result, roberta_result):
#     # Define thresholds for determining sentiment
#     vader_threshold = 0.5  # You can adjust this threshold based on your requirements
#     roberta_threshold = 0.5  # You can adjust this threshold based on your requirements

#     # Determine sentiment based on VADER score
#     if vader_result["vader_compound"] >= vader_threshold:
#         vader_sentiment = "POSITIVE"
#     elif vader_result["vader_compound"] <= -vader_threshold:
#         vader_sentiment = "NEGATIVE"
#     else:
#         vader_sentiment = "NEUTRAL"
#     # Determine sentiment based on RoBERTa score
#     if roberta_result["roberta_pos"] > roberta_result["roberta_neg"]:
#         roberta_sentiment = "POSITIVE"
#         roberta_score = roberta_result["roberta_pos"]
#     else:
#         roberta_sentiment = "NEGATIVE"
#         roberta_score = roberta_result["roberta_neg"]

#     # Return the combined sentiment and score
#     combined_sentiment = (
#         vader_sentiment if vader_sentiment != "NEUTRAL" else roberta_sentiment
#     )
#     combined_score = max(vader_result["vader_compound"], roberta_score)
#     return {"label": combined_sentiment, "score": combined_score}
