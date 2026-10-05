import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from textblob import TextBlob

# Set styling for plots
sns.set_theme(style="whitegrid")

# 1. Load the dataset
print("Loading dataset...")
df = pd.read_csv("1429_1.csv", low_memory=False)

# 2. Clean the data
df_clean = df.dropna(subset=["reviews.text", "reviews.rating"]).copy()
df_clean["name"] = df_clean["name"].str.replace(r"\r\n", " ", regex=True)


# 3. Define sentiment function
def get_sentiment(text):
  analysis = TextBlob(str(text))
  if analysis.sentiment.polarity > 0.1:
    return "Positive"
  elif analysis.sentiment.polarity < -0.1:
    return "Negative"
  else:
    return "Neutral"


print(
    "Running sentiment analysis on all reviews (this may take a few"
    " moments)..."
)
df_clean["predicted_sentiment"] = df_clean["reviews.text"].apply(get_sentiment)

# 4. Display Summary Metrics
print("\n--- Rating Distribution Summary ---")
rating_counts = df_clean["reviews.rating"].value_counts().sort_index()
print(rating_counts)

print("\n--- Predicted Sentiment Distribution ---")
sentiment_counts = df_clean["predicted_sentiment"].value_counts()
print(sentiment_counts)

# 5. Generate and Save Visualizations
print("\nGenerating charts...")

# Chart 1: Rating Distribution Bar Chart
plt.figure(figsize=(8, 5))
ax = sns.barplot(
    x=rating_counts.index, y=rating_counts.values, palette="Blues_d"
)
plt.title(
    "Amazon Product Reviews - Star Rating Distribution",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Star Rating", fontsize=12)
plt.ylabel("Number of Reviews", fontsize=12)

for p in ax.patches:
  height = int(p.get_height())
  ax.annotate(
      str(height),
      (p.get_x() + p.get_width() / 2.0, height),
      ha="center",
      va="center",
      xytext=(0, 5),
      textcoords="offset points",
  )

plt.tight_layout()
plt.savefig("rating_distribution.png", dpi=300)
plt.close()
print("Saved chart: rating_distribution.png")

# Chart 2: Predicted Sentiment Breakdown
plt.figure(figsize=(7, 5))
ax = sns.barplot(
    x=sentiment_counts.index, y=sentiment_counts.values, palette="Set2"
)
plt.title(
    "Predicted Sentiment Breakdown (TextBlob)",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Sentiment Category", fontsize=12)
plt.ylabel("Count", fontsize=12)

for p in ax.patches:
  height = int(p.get_height())
  ax.annotate(
      str(height),
      (p.get_x() + p.get_width() / 2.0, height),
      ha="center",
      va="center",
      xytext=(0, 5),
      textcoords="offset points",
  )

plt.tight_layout()
plt.savefig("sentiment_breakdown.png", dpi=300)
plt.close()
print("Saved chart: sentiment_breakdown.png")

print(
    "\nAnalysis complete! Check your folder for the generated PNG chart files."
)