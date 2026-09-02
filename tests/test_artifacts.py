"""Catch unfitted or dependency-incompatible saved artifacts before deployment."""

from pathlib import Path
import pickle
import unittest
import warnings

import numpy as np
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.utils.validation import check_is_fitted

from text_processing import transform_text


PROJECT_DIR = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Loading an old pickle under a different sklearn is not a valid test.
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            with (PROJECT_DIR / "vectorizer.pkl").open("rb") as file:
                cls.vectorizer = pickle.load(file)
            with (PROJECT_DIR / "model.pkl").open("rb") as file:
                cls.model = pickle.load(file)

    def test_both_artifacts_are_fitted_and_aligned(self):
        check_is_fitted(self.vectorizer)
        check_is_fitted(self.model)
        self.assertEqual(len(self.vectorizer.vocabulary_), self.model.n_features_in_)
        np.testing.assert_array_equal(self.model.classes_, [0, 1])

    def test_predictions_are_finite(self):
        messages = [
            "Can we meet at 10 tomorrow morning?",
            "Congratulations! You have won a free prize. Call now to claim.",
        ]
        vectors = self.vectorizer.transform([transform_text(text) for text in messages])
        np.testing.assert_array_equal(self.model.predict(vectors), [0, 1])
        probabilities = self.model.predict_proba(vectors)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)

    def test_preprocessing(self):
        self.assertEqual(transform_text("Hello! Running to the shops."), "hello run shop")
        self.assertEqual(transform_text("   !!! "), "")


if __name__ == "__main__":
    unittest.main()
