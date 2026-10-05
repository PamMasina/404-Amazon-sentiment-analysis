import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud

# Set plotting style
sns.set_theme(style="whitegrid")

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv("1429_1.csv", low_memory=False)

# 2. Clean Data
df_clean = df.dropna(subset=["reviews.text", "reviews.rating"]).copy()
df_clean["reviews.text"] = df_clean["reviews.text"].astype(str)

# 3. Sentiment Analysis Function
print("Running sentiment classification...")


def get_sentiment(text):
  analysis = TextBlob(text)
  if analysis.sentiment.polarity > 0.1:
    return "Positive"
  elif analysis.sentiment.polarity < -0.1:
    return "Negative"
  else:
    return "Neutral"


df_clean["sentiment"] = df_clean["reviews.text"].apply(get_sentiment)

# Calculate Summary Metrics
total_reviews = len(df_clean)
avg_rating = df_clean["reviews.rating"].mean()
sentiment_counts = df_clean["sentiment"].value_counts()
pct_positive = (
    sentiment_counts.get("Positive", 0) / total_reviews
) * 100
pct_negative = (
    sentiment_counts.get("Negative", 0) / total_reviews
) * 100

print("\n--- Summary Cards Metrics ---")
print(f"Total Reviews: {total_reviews:,}")
print(f"Average Rating: {avg_rating:.2f} / 5.0")
print(f"Positive Sentiment: {pct_positive:.1f}%")
print(f"Negative Sentiment: {pct_negative:.1f}%")

# 4. Pie Chart: Sentiment Split
print("\nGenerating Sentiment Split Pie Chart...")
plt.figure(figsize=(6, 6))
colors = ["#2ecc71", "#95a5a6", "#e74c3c"]
plt.pie(
    sentiment_counts.values,
    labels=sentiment_counts.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=colors[: len(sentiment_counts)],
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
plt.title(
    "Overall Sentiment Split (Positive / Neutral / Negative)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("dashboard_sentiment_pie.png", dpi=300)
plt.close()

# 5. Bar Chart: Sentiment by Star Rating
print("Generating Sentiment by Star Rating Bar Chart...")
plt.figure(figsize=(9, 5))
rating_sentiment = (
    pd.crosstab(df_clean["reviews.rating"], df_clean["sentiment"])
    .reindex(columns=["Positive", "Neutral", "Negative"])
    .fillna(0)
)
ax = rating_sentiment.plot(
    kind="bar", stacked=False, figsize=(9, 5), color=["#2ecc71", "#95a5a6", "#e74c3c"]
)
plt.title(
    "Sentiment Breakdown by Star Rating (1 - 5 Stars)",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Star Rating", fontsize=11)
plt.ylabel("Number of Reviews", fontsize=11)
plt.xticks(rotation=0)
plt.legend(title="Sentiment")
plt.tight_layout()
plt.savefig("dashboard_sentiment_by_rating.png", dpi=300)
plt.close()

# 6. Word Cloud: Most Common Words
print("Generating Word Cloud...")
all_text = " ".join(df_clean["reviews.text"])
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white",
    colormap="viridis",
    max_words=150,
).generate(all_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title(
    "Most Common Words Across All Reviews", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("dashboard_wordcloud.png", dpi=300)
plt.close()

# 7. Theme Extraction (Negative & Positive Themes)
print("Extracting Themes (Common Problems & Positive Highlights)...")


# Simple keyword-based theme classification for demonstration
def classify_negative_theme(text):
  t = text.lower()
  if any(w in t for w in ["return", "refund", "customer service", "return policy"]):
    return "Returns/Refunds"
  elif any(w in t for w in ["deliver", "shipping", "arrive", "late", "package", "box"]):
    return "Delivery Delays"
  elif any(w in t for w in ["site", "website", "app", "order", "checkout"]):
    return "Website/App Experience"
  elif any(w in t for w in ["battery", "charge", "screen", "break", "broke", "stop"]):
    return "Product Hardware Issue"
  else:
    return "General Dissatisfaction"


def classify_positive_theme(text):
  t = text.lower()
  if any(w in t for w in ["quality", "great product", "awesome", "sturdy", "durable"]):
    return "Product Quality"
  elif any(w in t for w in ["fast", "quick", "delivery", "shipping", "arrive early"]):
    return "Delivery Speed"
  elif any(w in t for w in ["price", "value", "cheap", "affordable", "cost"]):
    return "Price/Value"
  elif any(w in t for w in ["easy", "simple", "user friendly", "setup", "monitor"]):
    return "Ease of Use"
  else:
    return "General Satisfaction"


neg_df = df_clean[df_clean["sentiment"] == "Negative"].copy()
pos_df = df_clean[df_clean["sentiment"] == "Positive"].copy()

neg_df["theme"] = neg_df["reviews.text"].apply(classify_negative_theme)
pos_df["theme"] = pos_df["reviews.text"].apply(classify_positive_theme)

# Chart: Common Problems in Negative Reviews
neg_themes = neg_df["theme"].value_counts().head(5)
plt.figure(figsize=(8, 4))
ax = sns.barplot(x=neg_themes.values, y=neg_themes.index, palette="Reds_r")
plt.title(
    "Common Problems in Negative Reviews", fontsize=13, fontweight="bold"
)
plt.xlabel("Number of Reviews", fontsize=11)
plt.ylabel("Identified Theme", fontsize=11)
for p in ax.patches:
  ax.annotate(
      f"{int(p.get_width()):,}",
      (p.get_width(), p.get_y() + p.get_height() / 2.0),
      ha="left",
      va="center",
      xytext=(5, 0),
      textcoords="offset points",
  )
plt.tight_layout()
plt.savefig("dashboard_negative_themes.png", dpi=300)
plt.close()

# Chart: What Gets Positive Reviews
pos_themes = pos_df["theme"].value_counts().head(5)
plt.figure(figsize=(8, 4))
ax = sns.barplot(x=pos_themes.values, y=pos_themes.index, palette="Greens_r")
plt.title("What Gets Positive Reviews", fontsize=13, fontweight="bold")
plt.xlabel("Number of Reviews", fontsize=11)
plt.ylabel("Identified Theme", fontsize=11)
for p in ax.patches:
  ax.annotate(
      f"{int(p.get_width()):,}",
      (p.get_width(), p.get_y() + p.get_height() / 2.0),
      ha="left",
      va="center",
      xytext=(5, 0),
      textcoords="offset points",
  )
plt.tight_layout()
plt.savefig("dashboard_positive_themes.png", dpi=300)
plt.close()

print(
    "\nAll dashboard metrics and charts successfully generated and saved!"
)