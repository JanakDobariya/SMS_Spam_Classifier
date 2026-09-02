"""Text preprocessing shared by training and the Streamlit app."""

import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


_STEMMER = PorterStemmer()


@lru_cache(maxsize=1)
def _english_stopwords() -> set[str]:
    """Load the corpus once, downloading it on a fresh installation."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


def transform_text(text: str) -> str:
    """Normalize an SMS message in the same way for training and prediction."""
    tokens = nltk.word_tokenize(text.lower(), preserve_line=True)
    english_stopwords = _english_stopwords()
    filtered_tokens = (
        token
        for token in tokens
        if token.isalnum()
        and token not in english_stopwords
        and token not in string.punctuation
    )
    return " ".join(_STEMMER.stem(token) for token in filtered_tokens)
