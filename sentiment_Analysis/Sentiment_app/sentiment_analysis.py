from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import re
from googleapiclient.discovery import build
from nltk.sentiment import SentimentIntensityAnalyzer

youtubeAPIKey = "AIzaSyAi-UmIFNJuqFgivFmr2AuFTP3zPDA-ibQ"


def get_all_video_comments_with_ids(video_id):
    comments_with_ids = []
    next_page_token = None
    youtube = build("youtube", "v3", developerKey=youtubeAPIKey)

    while True:
        # Make a request to the YouTube API to retrieve comments
        response = (
            youtube.commentThreads()
            .list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                pageToken=next_page_token,
            )
            .execute()
        )

        # Extract comments with their IDs from the response
        for item in response["items"]:
            comment_id = item["snippet"]["topLevelComment"]["id"]
            comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments_with_ids.append({"id": comment_id, "text": comment_text})

        # Check if there are more pages
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments_with_ids


def extract_video_id(url):
    # Extract video ID from YouTube URL using regular expressions
    match = re.search(r"watch\?v=(\S+)", url)
    if match:
        return match.group(1)
    else:
        print("Invalid YouTube URL")
        return None


def poliarty_scores_roberta(example):
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
    # Perform sentiment analysis on the comments text
    # You can use any sentiment analysis method here
    # For example, you can use VADER or RoBERTa, or a combination of both

    # For demonstration purposes, let's assume a simple approach using VADER sentiment analyzer
    vader_analyzer = SentimentIntensityAnalyzer()
    vader_scores = vader_analyzer.polarity_scores(comments_text)

    # Define thresholds for determining sentiment
    vader_threshold = 0.5  # You can adjust this threshold based on your requirements

    # Determine sentiment based on VADER score
    if vader_scores["compound"] >= vader_threshold:
        sentiment_label = "POSITIVE"
    elif vader_scores["compound"] <= -vader_threshold:
        sentiment_label = "NEGATIVE"
    else:
        sentiment_label = "NEUTRAL"

    # Return the sentiment label and score
    return {"label": sentiment_label, "score": vader_scores["compound"]}


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
