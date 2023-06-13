import unittest
from spacy.language import Language
from spacy.tokens import Doc
from ..html_tokenizer import create_html_tokenizer, DEFAULT_REMOVE_TAGS, HTMLTokenizer


class TestHTMLTokenizer(unittest.TestCase):
    def setUp(self):
        # Set up a basic English language pipeline with our HTMLTokenizer
        self.nlp = Language()
        html_tokenizer_factory = create_html_tokenizer(DEFAULT_REMOVE_TAGS)
        self.nlp.tokenizer = html_tokenizer_factory(self.nlp)

    def test_html_tokenizer(self):
        # Create a basic HTML document
        html_doc = "<html><head><title>Test</title></head><body><p>Some text.</p></body></html>"

        # Parse the document with our tokenizer
        doc = self.nlp(html_doc)

        # Check that the tokens are what we expect
        expected_tokens = ["Some", "text", "."]
        actual_tokens = [token.text for token in doc]
        self.assertEqual(expected_tokens, actual_tokens)

    def test_remove_tags(self):
        # Create a HTML document with script tags
        html_doc = "<html><head><title>Test</title><script>var x = 10;</script></head><body><p>Some text.</p></body></html>"

        # Parse the document with our tokenizer
        doc = self.nlp(html_doc).text

        # Check that the script tag content is not in the tokens
        self.assertNotIn("var x = 10", doc)

    # Add more tests here for other functionalities.


if __name__ == "__main__":
    unittest.main()
