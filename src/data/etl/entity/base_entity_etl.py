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
        """
        Run after:
            1) all biomedical entities or owners are loaded
            2) all documents are loaded
            3) UMLS is loaded
        """
        await cls.link_to_documents()
        await cls.add_counts()
