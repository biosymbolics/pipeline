"""
Client stub for OpenAI
"""
import os
import logging

API_KEY = os.environ["OPENAI_API_KEY"]


class OpenAiApiClient:
    """
    Class for OpenAI API client
    """

    def __init__(self):
        self.client = None

    def get_client(self):
        """
        Returns client
        """
        # query_api = QueryApi(api_key=API_KEY)
        logging.error("Not yet implemented")
        return lambda x: x

    def __call__(self):
        if not self.client:
            self.client = self.get_client()
        return self.client


openai_client = OpenAiApiClient()
