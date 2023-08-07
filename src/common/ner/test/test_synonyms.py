import json
import unittest
from unittest.mock import patch, MagicMock
from common.ner.synonyms import SynonymStore  # replace with the actual module name


# class TestSynonymStore(unittest.TestCase):
#     @patch("redisearch.Client")
#     def setUp(self, mock_client):
#         self.mock_client = mock_client
#         self.synonym_store = SynonymStore("test_index")

#     def test_add_synonym(self):
#         term = "test_term"
#         canonical_id = "test_id"
#         metadata = {"meta1": "value1"}

#         self.synonym_store.add_synonym(term, canonical_id, metadata)
#         self.synonym_store.client.redis.hset.assert_called_once()

#         args, kwargs = self.synonym_store.client.redis.hset.call_args
#         self.assertEqual(args[0], f"term:{term}")

#         expected_mapping = {
#             b"term": bytes(term, "utf-8"),
#             b"canonical_id": bytes(canonical_id, "utf-8"),
#             b"metadata": bytes(json.dumps(metadata), "utf-8"),
#         }
#         self.assertEqual(kwargs["mapping"], expected_mapping)

#     @patch("common.utils.string.get_id")
#     def test_get_synonym(self, mock_get_id):
#         term = "test_term"
#         canonical_id = "test_id"
#         metadata = {"meta1": "value1"}
#         mock_get_id.return_value = term

#         expected_result = {
#             "term": term,
#             "canonical_id": canonical_id,
#             "metadata": metadata,
#         }
#         self.synonym_store.client.redis.hgetall.return_value = {
#             b"term": bytes(term, "utf-8"),
#             b"canonical_id": bytes(canonical_id, "utf-8"),
#             b"metadata": bytes(json.dumps(metadata), "utf-8"),
#         }

#         result = self.synonym_store.get_synonym(term)
#         self.synonym_store.client.redis.hgetall.assert_called_once_with(f"term:{term}")
#         self.assertEqual(result, expected_result)


# # Add more test cases here for other methods

# if __name__ == "__main__":
#     unittest.main()
