"""
Abstract document ETL class
"""
from abc import abstractmethod
import logging

from system import initialize

initialize()


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseEntityEtl:
    def __init__(self, document_type: str):
        self.document_type = document_type

    async def copy_all(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    async def pre_doc_finalize():
        pass

    @staticmethod
    async def add_counts():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    async def link_to_documents():
        raise NotImplementedError

    @classmethod
    async def post_doc_finalize(cls):
        raise NotImplementedError
