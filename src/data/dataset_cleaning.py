import unicodedata
import re

import pandas as pd


def word_count(text):
    words = text.split()
    return len(words)

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # normalize unicode (e.g. curly quotes, accented chars)
    text = unicodedata.normalize("NFKC", text)

    # remove newlines and carriage returns
    text = text.replace("\n", " ").replace("\r", " ")

    # remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

if __name__ == "__main__":
    data = pd.read_csv("raw_data/AI_Human.csv")
    data["generated"] = data["generated"].astype("int")

    data["word_count"] = data["text"].apply(word_count)
    data = data[data["word_count"] > 0]
    data["text"] = data["text"].apply(clean_text)

    duplicate_texts = (
        data
        .groupby("text")
        .size()
        .reset_index(name="n_rows")
        .query("n_rows > 1")
        .sort_values("n_rows", ascending=False)
    )

    print("Number of duplicated text strings:", len(duplicate_texts))

    clean_data = data.drop_duplicates(subset="text", keep='first').copy()

    print("Initial number of rows:", len(data))
    print("Number of rows after removing duplicate texts:", len(clean_data))
    print("Number of rows removed:", len(data) - len(clean_data))
    clean_data = clean_data[['text', 'generated']]
    clean_data.to_csv('clean_data/kaggle_dataset.csv', index=False)