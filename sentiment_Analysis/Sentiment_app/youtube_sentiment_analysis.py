# %%


# %%
from googleapiclient.discovery import build
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

import nltk

# %%
youtubeAPIKey = "AIzaSyAIuSegS5UllyaWInMpcNDZkHLxErHxwIg"
youtube = build('youtube','v3',developerKey=youtubeAPIKey)

# %%
import re

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


# Example URL
video_url = "https://www.youtube.com/watch?v=3rVF1pT-jRY"

# Extract video ID from URL
video_id = extract_video_id(video_url)

if video_id:
    # Get all comments with IDs for the video
    all_comments_with_ids = get_all_video_comments_with_ids(video_id)

    # Now you have all comments with their corresponding IDs
    print("Comments:")
    for comment_info in all_comments_with_ids:
        print("ID:", comment_info["id"])
        print("Text:", comment_info["text"])
else:
    print("Error extracting video ID from URL")

# %%
comment_dict = {
    comment_info["id"]: comment_info["text"] for comment_info in all_comments_with_ids
}

# Print head 5 comments
print("Head 5 comments:")
for comment_id, comment_text in list(comment_dict.items())[:5]:
    print("ID:", comment_id)
    print("Text:", comment_text)

# Print tail 5 comments
print("\nTail 5 comments:")
for comment_id, comment_text in list(comment_dict.items())[-5:]:
    print("ID:", comment_id)
    print("Text:", comment_text)

# %%
comment_text = "UgxK4UB3mhQ7Yn4POc14AaABAg"  # Example comme ID yntou want to retrieve
example = comment_dict.get(
    comment_text
)  # Retrieve the comment text corresponding to the comment ID
print(example)

# %%
# Retrieve the comment text corresponding to the 40th comment
example = list(comment_dict.values())[39]  # Assuming comments are 0-indexed
print(example)

# %%
import nltk
tokens = nltk.word_tokenize(example)
tokens[:10]

# %%
tagged = nltk.pos_tag(tokens)
tagged[:10]
# ntlk pos_tag speech tagging

# %%
import nltk

nltk.download("maxent_ne_chunker")
nltk.download("words")

# %%
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()
# it will form group of entiters into chunks

# %%
import nltk

nltk.download("vader_lexicon")

# %%
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()

# %%
sia.polarity_scores(example)

# %%

# %%
# Import the necessary libraries
from tqdm.notebook import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer

# Create SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# %%
comment_dict

# %%
res = {}
for comment_id, comment_text in tqdm(comment_dict.items(), total=len(comment_dict)):
    # Process each comment here
    res[comment_id] = sia.polarity_scores(comment_text)

# %%
import nltk

nltk.download("vader_lexicon")

# %%
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()

# %%
sia.polarity_scores(example)

# %%
# Import the necessary libraries
from tqdm.notebook import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer

# Create SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# %%
# # Convert the dictionary to a DataFrame
# comment_df = pd.DataFrame(list(comment_dict.items()), columns=["Id", "Text"])

# # Merge vaders DataFrame with comment_df DataFrame
# vaders = vaders.merge(comment_df, how="left", on="Id")
# Convert the dictionary to a DataFrame
comment_df = pd.DataFrame(list(comment_dict.items()), columns=["Id", "Text"])

# Merge vaders DataFrame with comment_df DataFrame, specifying suffixes for overlapping columns
vaders = vaders.merge(comment_df, how="left", on="Id", suffixes=('_vaders', '_comment'))


# %%
comment_df

# %%
res = {}
for i, row in tqdm(comment_df.iterrows(), total=len(comment_df)):
    text = row["Text"]
    myid = row["Id"]
    res[myid] = sia.polarity_scores(text)

# %%
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={"index": "Id"})
vaders = vaders.merge(comment_df, how="left")

# %%
# now we have sentiment score and metadata
vaders.head()

# %%
ax = sns.barplot(data=vaders, x="Id", y="compound")
# ax.set_title("Compound Score by Youtube link view")
# plt.show()

# %%
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x="Id", y="pos", ax=axs[0])
sns.barplot(data=vaders, x="Id", y="pos", ax=axs[1])
sns.barplot(data=vaders, x="Id", y="pos", ax=axs[2])
axs[0].set_title("Positive")
axs[1].set_title("Neutral")
axs[2].set_title("Negative")
plt.tight_layout()
plt.show()





# %%
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

# %%


# %%
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# %%
# VADER result on example
print(example)
sia.polarity_scores(example)

# %%
# Run for Roberta Model
encoded_text = tokenizer(example, return_tensors="pt")
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scored = softmax(scores)
scores_dict = {
    "roberta_neg": scores[0],
    "roberta_neu": scores[1],
    "roberta_pos": scores[2],
}
print(scores_dict)

# %%
def poliarty_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors="pt")
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores_dict = {
        "roberta_neg": scores[0],
        "roberts_neu": scores[1],
        "roberta_pos": scores[2],
    }
    return scores_dict

# %%
res = {}
for i, row in tqdm(comment_df.iterrows(), total=len(comment_df)):
    try:
        text = row["Text"]
        myid = row["Id"]
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value

        roberta_result = poliarty_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f"Broke of id {myid}")

# %%
both

# %% [markdown]
# Vader Result

# %%
vader_result

# %%
roberta_result

# %%
result_df = pd.DataFrame(res).T
result_df = result_df.reset_index().rename(columns={"index": "Id"})
results_df = result_df.merge(comment_df, how="left")

# %%
results_df.head()

# %%
results_df.columns

# %% [markdown]
# Combine and Compare

# %%
# sns.pairplot(
#     data=results_df,
#     vars=[
#         "vader_neg",
#         "vader_neu",
#         "vader_pos",
#         "roberta_neg",
#         "roberts_neu",
#         "roberta_pos",
#     ],
#     hue="Id",
#     palette="tab10",
# )
# plt.show()

# %%
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


# # Example usage:
# vader_result = {
#     "vader_neg": 0.0,
#     "vader_neu": 0.27,
#     "vader_pos": 0.73,
#     "vader_compound": 0.6597,
# }

# roberta_result = {
#     "roberta_neg": -0.9672584,
#     "roberts_neu": 0.3386652,
#     "roberta_pos": 0.7277422,
# }


combined_result = evaluate_sentiment(vader_result, roberta_result)
print(combined_result)


