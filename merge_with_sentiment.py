import pandas as pd
import numpy as np

def main():
    processed = pd.read_csv("./processed_reviews_without_sentiment.csv")
    sentiment = pd.read_csv("./sentiment.csv")
    sentiment = sentiment[["asin", "reviewerID", "sentiment"]]
    processed = processed.merge(sentiment, on=["asin", "reviewerID"])
    processed = processed.dropna()
    processed = processed.reset_index(drop=True)

    processed.to_csv("./processed_reviews.csv", index=False)

if __name__ == "__main__":
    main()