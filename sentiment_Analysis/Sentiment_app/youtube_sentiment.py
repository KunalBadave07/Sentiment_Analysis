from googleapiclient.discovery import build
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from tqdm.notebook import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

plt.style.use("ggplot")

# %%
youtubeAPIKey = "AIzaSyAIuSegS5UllyaWInMpcNDZkHLxErHxwIg"
youtube = build('youtube','v3',developerKey=youtubeAPIKey)

# Create a YouTube API client
youtube = build("youtube", "v3", developerKey=youtubeAPIKey)

def get_all_video_comments_with_ids(video_id):
    comments_with_ids = []
    next_page_token = None

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
    encoded_text = tokenizer(example, return_tensors="pt")
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores_dict = {
        "roberta_neg": scores[0],
        "roberta_neu": scores[1],
        "roberta_pos": scores[2],
    }
    return scores_dict


def evaluate_sentiment(vader_result, roberta_result):
    # Define thresholds for determining sentiment
    vader_threshold = 0.5  # You can adjust this threshold based on your requirements
    roberta_threshold = 0.5  # You can adjust this threshold based on your requirements

    # Determine sentiment based on VADER score
    if vader_result["vader_compound"] >= vader_threshold:
        vader_sentiment = "POSITIVE"
    elif vader_result["vader_compound"] <= -vader_threshold:
        vader_sentiment = "NEGATIVE"
    else:
        vader_sentiment = "NEUTRAL"
    # Determine sentiment based on RoBERTa score
    if roberta_result["roberta_pos"] > roberta_result["roberta_neg"]:
        roberta_sentiment = "POSITIVE"
        roberta_score = roberta_result["roberta_pos"]
    else:
        roberta_sentiment = "NEGATIVE"
        roberta_score = roberta_result["roberta_neg"]

    # Return the combined sentiment and score
    combined_sentiment = (
        vader_sentiment if vader_sentiment != "NEUTRAL" else roberta_sentiment
    )
    combined_score = max(vader_result["vader_compound"], roberta_score)
    return {"label": combined_sentiment, "score": combined_score}


def get_combined_sentiment(youtube_url):
    video_id = extract_video_id(youtube_url)
    if video_id:
        # Get all comments with IDs for the video
        all_comments_with_ids = get_all_video_comments_with_ids(video_id)
        
        # Create SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()

        # Retrieve the sentiment model
        MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)

        # Convert the comments to a dictionary
        comment_dict = {
            comment_info["id"]: comment_info["text"] for comment_info in all_comments_with_ids
        }

        # Analyze comments with VADER and RoBERTa models
        res = {}
        for i, row in tqdm(comment_dict.items(), total=len(comment_dict)):
            try:
                text = row["Text"]
                myid = row["Id"]
                vader_result = sia.polarity_scores(text)
                    # res[comment_id] = sia.polarity_scores(comment_text)
                vader_result_rename = {}
                for key, value in vader_result.items():
                    vader_result_rename[f"vader_{key}"] = value

                roberta_result = poliarty_scores_roberta(text)
                both = {**vader_result_rename, **roberta_result}
                res[myid] = both
            except RuntimeError:
                print(f"Broke of id {myid}")

        # Create a DataFrame
        result_df = pd.DataFrame(res).T
        result_df = result_df.reset_index().rename(columns={"index": "Id"})
        results_df = result_df.merge(comment_df, how="left")

        # Combine and Compare
        def evaluate_sentiment(vader_result, roberta_result):
            # Define thresholds for determining sentiment
            vader_threshold = 0.5  # You can adjust this threshold based on your requirements
            roberta_threshold = 0.5  # You can adjust this threshold based on your requirements

            # Determine sentiment based on VADER score
            if vader_result["vader_compound"] >= vader_threshold:
                vader_sentiment = "POSITIVE"
            elif vader_result["vader_compound"] <= -vader_threshold:
                vader_sentiment = "NEGATIVE"
            else:
                vader_sentiment = "NEUTRAL"
            # Determine sentiment based on RoBERTa score
            if roberta_result["roberta_pos"] > roberta_result["roberta_neg"]:
                roberta_sentiment = "POSITIVE"
                roberta_score = roberta_result["roberta_pos"]
            else:
                roberta_sentiment = "NEGATIVE"
                roberta_score = roberta_result["roberta_neg"]

            # Return the combined sentiment and score
            combined_sentiment = (
                vader_sentiment if vader_sentiment != "NEUTRAL" else roberta_sentiment
            )
            combined_score = max(vader_result["vader_compound"], roberta_score)
            return {"label": combined_sentiment, "score": combined_score}
        
        combined_result = evaluate_sentiment(vader_result, roberta_result)
        if combined_result['label'] == 'POSITIVE':
            return 1
        elif combined_result['label'] == 'NEGATIVE':
            return -1
        else:
            return 0
    else:
        print("Error extracting video ID from URL")

# Example URL
urltext = input("Enter url here : ")
# video_url = "https://www.youtube.com/watch?v=3rVF1pT-jRY"
video_url = urltext
combined_sentiment = get_combined_sentiment(video_url)
print("Combined Sentiment:", combined_sentiment)
