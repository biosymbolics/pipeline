"""
Class for cached web page reading
"""
import logging
import os
import pathlib
import requests
import html2text
import requests_random_user_agent
from llama_index.readers.schema.base import Document

from common.utils.url import url_to_filename


class CachedWedPageReader:
    """
    like SimpleWebPageReader but with caching
    TODO: implement html_to_text bool
    """

    def __init__(self, storage_dir: str, namespace: str):
        self.storage_dir = storage_dir
        self.namespace = namespace

    def __get_doc_dir(self) -> str:
        doc_dir = f"{self.storage_dir}/{self.namespace}"
        pathlib.Path(doc_dir).mkdir(parents=True, exist_ok=True)
        return doc_dir

    def __get_file_location(self, url) -> str:
        file_name = url_to_filename(url)
        return f"{self.__get_doc_dir()}/{file_name}"

    def __cache_page(self, url: str, page_text: str):
        file_location = self.__get_file_location(url)
        with open(file_location, "w") as file:
            file.write(page_text)

    def __get_document(self, url: str):
        file_location = self.__get_file_location(url)
        if os.path.isfile(file_location):
            # load from the file if present
            with open(file_location, "r") as file:
                document = Document(file.read())
            return document

        # else pull webpage
        # use requests-random-user-agent ?
        response = requests.get(url, headers=None).text
        response = html2text.html2text(response)
        document = Document(response)
        return document

    def load_data(self, urls: list[str]):
        """
        Load data
        """
        if not isinstance(urls, list):
            raise ValueError("urls must be a list of strings.")

        documents = []
        for url in urls:
            logging.debug("Getting url %s", url)
            document = self.__get_document(url)
            documents.append(document)
            self.__cache_page(url, document.text)

        return documents
