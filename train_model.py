"""Train and save the SMS spam classifier artifacts."""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from text_processing import transform_text


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "Data" / "spam.csv"


def load_data() -> tuple[pd.Series, pd.Series]:
    data = pd.read_csv(DATA_PATH, encoding="latin-1", usecols=["v1", "v2"])
    data = data.rename(columns={"v1": "target", "v2": "text"}).drop_duplicates()
    targets = data["target"].map({"ham": 0, "spam": 1})
    messages = data["text"].apply(transform_text)
    return messages, targets


def main() -> None:
    messages, targets = load_data()
    train_messages, test_messages, train_targets, test_targets = train_test_split(
        messages,
        targets,
        test_size=0.2,
        random_state=2,
        stratify=targets,
    )

    evaluation_vectorizer = TfidfVectorizer(max_features=3000)
    train_vectors = evaluation_vectorizer.fit_transform(train_messages)
    test_vectors = evaluation_vectorizer.transform(test_messages)
    evaluation_model = MultinomialNB().fit(train_vectors, train_targets)
    predictions = evaluation_model.predict(test_vectors)

    print(f"Accuracy:  {accuracy_score(test_targets, predictions):.4f}")
    print(f"Precision: {precision_score(test_targets, predictions):.4f}")

    # Refit on all available data before creating the production artifacts.
    vectorizer = TfidfVectorizer(max_features=3000)
    all_vectors = vectorizer.fit_transform(messages)
    model = MultinomialNB().fit(all_vectors, targets)

    with (PROJECT_DIR / "vectorizer.pkl").open("wb") as file:
        pickle.dump(vectorizer, file)
    with (PROJECT_DIR / "model.pkl").open("wb") as file:
        pickle.dump(model, file)

    print(f"Saved fitted artifacts for {len(messages)} unique messages.")


if __name__ == "__main__":
    main()
