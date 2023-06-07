"""
Class for cached web page reading
"""
import logging
import os
import pathlib
import requests
import html2text

import requests_random_user_agent  # necessary
from llama_index.readers.schema.base import Document

from types.indices import NamespaceKey
from common.utils.file import save_as_file
from common.utils.namespace import get_namespace
from common.utils.url import url_to_filename


class CachedWedPageReader:
    """
    like SimpleWebPageReader but with caching
    TODO: implement html_to_text bool
    example use:
        documents = CachedWedPageReader("./sec_docs", "pfe").load_data([url])
        index = GPTListIndex.from_documents(documents)
    """

    def __init__(self, storage_dir: str, namespace_key: NamespaceKey):
        self.storage_dir = storage_dir
        self.namespace = get_namespace(namespace_key)

    def __get_doc_dir(self) -> str:
        doc_dir = f"{self.storage_dir}/{self.namespace}"
        pathlib.Path(doc_dir).mkdir(parents=True, exist_ok=True)
        return doc_dir

    def __get_file_location(self, url) -> str:
        file_name = url_to_filename(url)
        return f"{self.__get_doc_dir()}/{file_name}"

    def __cache_page(self, url: str, page_text: str):
        file_location = self.__get_file_location(url)
        save_as_file(page_text, file_location)

    def __get_document(self, url: str):
        file_location = self.__get_file_location(url)
        if os.path.isfile(file_location):
            # load from the file if present
            with open(file_location, "r", encoding="utf-8") as file:
                document = Document(file.read())
            return document

        # else pull webpage
        response = requests.get(url, headers=None, timeout=10000).text
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
            if document.text:
                self.__cache_page(url, document.text)

        return documents
