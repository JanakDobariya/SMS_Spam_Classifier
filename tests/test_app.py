"""Check startup and the main classification paths using the saved model."""

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class ClassifierSmokeTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        self.assertFalse(self.app.exception)

    def classify(self, message):
        self.app.text_area[0].set_value(message)
        self.app.button[0].click().run()
        self.assertFalse(self.app.exception)

    def test_initial_page(self):
        self.assertEqual(self.app.title[0].value, "Email/SMS Spam Classifier")
        self.assertEqual(len(self.app.success), 0)
        self.assertEqual(len(self.app.error), 0)

    def test_empty_message(self):
        self.classify("   ")
        self.assertEqual(self.app.warning[0].value, "Please enter a message first.")

    def test_legitimate_message(self):
        self.classify("Can we meet at 10 tomorrow morning?")
        self.assertEqual(self.app.success[0].value, "Ham")

    def test_spam_message(self):
        self.classify("Congratulations! You have won a free prize. Call now to claim.")
        self.assertEqual(self.app.error[0].value, "Spam")


if __name__ == "__main__":
    unittest.main()
