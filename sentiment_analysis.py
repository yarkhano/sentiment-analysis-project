import pandas as pd


def encode(rating: int) -> str:
    if rating <= 2:
        return "Negative"
    if rating == 3:
        return "Neutral"
    return "Positive"


dataset = pd.read_csv("product_reviews_mock_data.csv")
dataset.drop(columns=["ReviewID", "ProductID", "UserID", "ReviewDate"], inplace=True)

label_map = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2,
}

dataset["sentiment"] = dataset["Rating"].apply(encode)
dataset["label"] = dataset["sentiment"].map(label_map)

print(dataset.head(5))
